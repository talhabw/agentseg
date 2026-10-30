# Plan: SAM 3 tabanlı, ajanın kullandığı etiketleme sistemi

30 Ekim 2026. Hedef: ajana "şu torpedoyu etiketle" denince ajanın sadece `autoseg` CLI'ı
ve onun skill'ini kullanarak diziyi baştan sona etiketlemesi, export etmesi ve kısa bir
rapor vermesi. Ajan ancak sistem bir sorun bildirdiğinde uğraşır.

## Sabit kararlar

- **Tek model: SAM 3** (`sam_server --models sam3,image`). Image kafası maske üretir
  (kutu + pozitif/negatif tık), video kafası onu taşır (`add_mask` + `propagate_in_video`).
  SAM 3.1 ve text yolu kapsam dışı.
- **Keyframe kaynak, propagate türev.** Keyframe = ajanın/insanın onayladığı maske
  (`source != "propagated"` olan, track'e bağlı annotation). Propagate sonuçları
  (`source = "propagated"`) her an keyframe'lerden yeniden üretilebilir. SAM oturumu
  sadece önbellek; keyframe silinince oturum yeniden kurulur.
- **Durumun sahibi backend.** CLI backend'in REST API'sine konuşan ince bir istemci;
  frontend aynı projeyi canlı gösterir. sam_server değişmeden model düğümü olarak kalır.
- **Ajan arayüzü: `autoseg` CLI + bir skill.** MCP yok. CLI her komutta JSON basar,
  görselleri PNG dosyası olarak yazıp yolunu verir (ajan dosyayı okuyarak görür).
- **DINO artık tracker değil:** sağlık sinyali (keyframe'e benzerlik) ve kayıp nesneyi
  yeniden bulma (fallback). Fallback dolu bir maskenin üzerine asla otomatik yazmaz.
- Çevrimdışı etiketleme: dizi başına birkaç dakika kabul edilebilir.

## Genel akış

```
WP0 hazırlık ─► WP1 motor ─► WP2 sağlık/issue ─► WP3 oracle ajan + ölçüm ─┐
                                                                          ├─► WP4 backend ─► WP5 CLI ─► WP6 skill ─► WP7 ajan denemesi
                                      WP3'te stres dizileri (WP0.3) ──────┘                      │
                                                                     WP8 fallback (WP3/WP7 gerektirirse) ◄──┘
                                                                     WP9 frontend (paralel, opsiyonel)
```

Kapı noktaları: WP3'ün kabul kriteri sağlanmadan WP4'e geçilmez (motor ve sinyaller
ajansız kanıtlanır). WP7, oracle sonuçlarıyla karşılaştırılarak değerlendirilir.

---

## WP0 · Hazırlık (S)

1. `scripts/sam3_video_eval.py` commit edilir (şu an untracked), önceki 13 koşunun
   `summary.json`'ları `docs/plans/results/2026-10-30/` altına özet tablo olarak alınır.
2. sam_server çalıştırma tarifi sabitlenir: `--models sam3,image`, port 8770,
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. `sam_server/README.md` güncellenir.
3. **Değerlendirme verisi.** Elde tek GT dizisi var (torpedo, 526 kare, 248 benzersiz).
   - Gerçek ek diziler: kullanıcıdan 2–4 etiketli dizi (kadrajdan çıkıp giren nesne,
     örtülme, iki nesne, sınıf 7). **Açık soru.**
   - Gelene kadar torpedo dizisinden türetilmiş stres dizileri (GT dönüştürülerek):
     - `sub4`: her 4. kare (büyük hareket, düşük fps)
     - `exit`: kayan kırpma penceresi; nesne ~40 kare kadrajdan çıkıp geri gelir
     - `occl`: 20 karelik aralıkta nesnenin %60–100'ünü örten yapay yama
     - `mid`: ortadan başlatma (iki yön)
   - Script: `scripts/make_stress_seqs.py`. Çıktı: `<int>.jpg` kareler + YOLO label'ları.

Kabul: eval script'i main'de; 4 stres dizisi üretilmiş ve GT overlay'leri gözle kontrol edilmiş.

## WP1 · Takip motoru (M)

Backend'den ve oracle'dan aynen kullanılacak saf Python kütüphanesi: `ml/sam3_video/`.

- `client.py`: sam_server istemcisi (session aç/kapat, `add_mask`, `remove_object`,
  stream propagate, `cancel_propagation`, image `predict_inst` kutu + tık). RLE codec'i
  `sam_server.codec`'ten paylaşılır (path dependency ya da küçük kopya; karar WP1'de).
- `frames.py`: kare listesinden `<int>.jpg` symlink klasörü üretir, indeks eşlemesini tutar.
- `engine.py`: `TrackEngine`
  - durum: `tracks[track_id] → keyframes{frame: mask | ABSENT}`
  - `propagate(track, range)`: keyframe'leri `add_mask` ile verir, en erken keyframe'den
    ileri ve geri propagate eder, kare kare maske yayar (generator).
  - `add_keyframe(track, frame, mask)`: artımlı yol; o kareden ileri, bir sonraki
    keyframe'e kadar yeniden propagate.
  - `remove_keyframe(...)`: oturumu sıfırlar (`remove_object` + kalan keyframe'leri yeniden
    ekleme), etkilenen aralığı yeniden üretir.
  - `mark_absent(track, frames)`: bu karelerde nesne yok kabul edilir; issue/fallback
    tetiklenmez. (SAM'e boş maske verilip verilemediği WP1'de denenir; olmazsa sadece
    motor seviyesinde maskeleme.)
  - Çok track = aynı oturumda farklı `obj_id`.
- Uzun diziler: pencereleme (ör. 500 kare, örtüşmeli, pencere sınırında son maske keyframe
  olarak taşınır). İlk sürümde arayüz hazır, uygulama WP8'e kadar ertelenebilir.

Kabul: birim testleri (sam_server sahte istemcisi ile, GPU'suz); gerçek sunucuyla torpedo
dizisinde E koşusu sonucu (ort. IoU ≈ 0.963) yeniden üretilir; keyframe silme sonrası
sonuç, o keyframe hiç eklenmemiş koşuyla aynı.

## WP2 · Sağlık kontrolü ve issue'lar (M)

`ml/sam3_video/health.py`. Her kare için ucuz sinyaller, sonra durum ve issue gruplama.

| Sinyal | Hesap |
|---|---|
| boş maske | arka arkaya boş kare sayısı |
| alan oranı | önceki kareye ve en yakın keyframe'e göre (alan trendi `propagate.update_area_trend` mantığıyla) |
| merkez sıçraması | kutu köşegenine oranlı |
| parçalanma | bileşen sayısı, en büyük bileşenin payı |
| kenara değme | bilgi amaçlı; alan düşüşünü açıklar |
| DINO benzerliği | maskeli bölge embedding'i vs keyframe'ler (`ml/embed.py`), her k karede bir veya şüpheli karelerde |

- Durum: `ok | review | lost`. Eşikler bir `HealthConfig`'te, varsayılanlar WP3'te ölçülür.
- **Issue** = arka arkaya `review/lost` kareler (küçük boşluklar birleştirilir).
  Alanlar: `track, start, end, worst_frame, kind (lost|scope|jump|fragment|drift), signals`.
- Ajana issue başına tek bir "en kötü kare" önerilir.

Kabul: sinyaller torpedo + stres dizilerinde frame bazında CSV'ye yazılıyor; testler.

## WP3 · Oracle ajan ve ölçüm (M) — kapı

`scripts/oracle_agent.py`: motor + sağlık modülünü kullanan, kararlarını GT'den veren
simüle ajan. Gerçek ajanın yapacağı döngünün aynısı, tek fark gözü GT.

1. Prompt karesinde GT kutusundan image maske → keyframe.
2. Propagate → issue listesi.
3. Her issue için en kötü karede GT kutusundan yeni maske → keyframe (artımlı propagate).
   Kötü keyframe senaryosunda (F deneyi) `remove_keyframe` dener.
4. Issue kalmayana ya da bütçe (ör. 10 müdahale) dolana kadar tekrar.

Metrikler (dizi başına, tek JSON + HTML rapor):
- son ort. IoU, isabet oranı (IoU ≥ 0.5) ve iyi oran (IoU ≥ 0.8)
- müdahale sayısı / 100 kare
- issue yakalama: IoU < 0.8 olan karelerin yüzde kaçı bir issue içinde (recall), issue
  karelerinin yüzde kaçı gerçekten kötü (precision)
- toplam süre

Eşik ayarı: sinyal eşikleri recall ≥ 0.9 ve makul precision verecek şekilde seçilir,
`HealthConfig` varsayılanlarına yazılır.

Kabul (kapı):
- torpedo: ort. IoU ≥ 0.95, ≤ 1 müdahale / 100 kare
- stres dizileri: ort. IoU ≥ 0.90; `exit` ve `occl` dizilerinde kayıp issue olarak yakalanıyor
- issue recall ≥ 0.9
Sağlanamazsa: hangi dizide neden olmadığı raporlanır ve WP8 (fallback) öne çekilir.

## WP4 · Backend entegrasyonu (L)

- **Image segmentasyon sam_server'a taşınır.** `ml/segment.py` arkasına sam_server
  istemcisi; yerel SAM yükleme seçenek olarak kalır. GPU'da tek kopya.
- **VideoTrackJob** (`ml/tracking_job.py` iskeleti: event akışı, durdur/devam, snapshot).
  - Proje açılınca symlink kare klasörü üretilir (`order_index` sırasıyla).
  - Keyframe'ler DB'den okunur, motor çalışır, her kare `Annotation(source="propagated",
    track_id, status=approved|needs_review)` olarak yazılır; maske → polygon/RLE mevcut
    `core/masks.py` ile.
  - Yeniden üretimde sadece o track'in `propagated` annotation'ları değişir.
  - Issue'lar job event'i olarak yayınlanır ve DB'de saklanır (yeni küçük tablo `issues`).
- **API** (`backend/api/video_tracks.py`):
  - `POST /video/segment` (önizleme, kaydetmez; preview id döner)
  - `POST /video/tracks` (preview'dan track + ilk keyframe)
  - `POST /video/tracks/{id}/propagate` (job başlatır)
  - `GET  /video/tracks/{id}/issues`
  - `POST /video/tracks/{id}/keyframes` (refine: kutu/tık → keyframe + artımlı propagate)
  - `DELETE /video/tracks/{id}/keyframes/{frame}`
  - `POST /video/tracks/{id}/absent`
  - `GET  /video/render/frame`, `GET /video/render/sheet` (overlay PNG)
- Mevcut `/api/track_jobs` event akışı yeniden kullanılır.

Kabul: API testleri (sahte sam_server); gerçek sunucuyla WP3'teki oracle'ın aynı sonucu
backend üzerinden üretmesi (`oracle_agent.py --via-backend` modu).

## WP5 · `autoseg` CLI (M)

`pyproject.toml` → `[project.scripts] autoseg = "cli.main:app"`. Backend'e HTTP istemcisi.
Her komut stdout'a tek JSON basar; görseller `--out` klasörüne (varsayılan
`.autoseg/views/`) PNG olarak yazılır, JSON'da yolu verilir. Hata = sıfırdan farklı çıkış
kodu + JSON `error`.

```
autoseg status                                   # backend + sam_server + proje durumu
autoseg project open <dir>                       # ya da create; kare sayısı, boyut
autoseg labels [--add NAME]
autoseg view <frame> [--track T] [--grid]        # overlay PNG; --grid koordinat ızgarası
autoseg sheet [--track T] [--every N | --frames a,b,c]
autoseg segment <frame> --box x0,y0,x1,y1 [--pos x,y ...] [--neg x,y ...]   # önizleme PNG + preview id
autoseg track create --label L --preview P
autoseg track propagate T [--wait]               # bitince özet + issue listesi
autoseg track issues T
autoseg track refine T <frame> (--preview P | --box ... [--pos ...] [--neg ...])
autoseg track keyframes T [--rm FRAME]
autoseg track absent T <a-b>
autoseg track list
autoseg export [--format yolo-seg|bbox] [--out DIR]
```

- Koordinatlar piksel, xyxy; JSON'da da aynı kural. `view --grid` ajan için 50 px ızgara
  çizer (kutu tahmini kolaylaşsın).
- Uzun işler `--wait` ile bloklar ve ilerlemeyi stderr'e basar; JSON sonuç stdout'ta.

Kabul: komut başına test (sahte backend); gerçek sistemde torpedo dizisi CLI ile elle
baştan sona etiketlenebiliyor; `autoseg --help` ve her alt komut `--help` eksiksiz.

## WP6 · Skill (S)

`.claude/skills/autoseg-label/SKILL.md` (+ gerekirse `reference.md`). İçerik:

- Ne zaman kullanılır: "X'i etiketle / takip et" istekleri.
- Ön koşul: `autoseg status`; sam_server/backend kapalıysa nasıl açılır.
- Standart akış:
  1. `project open`, nesnenin göründüğü bir kareyi bul (`sheet --every N`).
  2. `view --grid` → kutu tahmini → `segment` → önizlemeye bak. Kapsam yanlışsa
     pozitif/negatif tıkla düzelt (ör. ayaklara negatif). Onaylayınca `track create`.
  3. `track propagate --wait`.
  4. Her issue için en kötü kareye `view`, karar: `refine` / `absent` / yok say (gerekçeli).
     Kötü keyframe şüphesinde `keyframes --rm`.
  5. Issue kalmayınca `sheet --every N` ile son kontrol, `export`.
  6. Rapor: track'ler, keyframe sayısı, kalan/yok sayılan issue'lar, kontak sayfası yolu.
- Kurallar: önizlemeye bakmadan keyframe yapma; aynı issue'ya 2 başarısız düzeltmeden
  sonra kullanıcıya sor; bütçe: müdahale başına bir görsel, toplam ~N görsel.
- Kısa örnek oturum (gerçek CLI çıktılarıyla, WP7'nin ilk koşusundan alınır).

Kabul: skill, WP7'de ajanın ek açıklama almadan akışı tamamlamasını sağlıyor.

## WP7 · Uçtan uca ajan denemesi (M)

Kör deneme: ajan GT'yi görmez.

- Düzen: `scripts/agent_trial.sh <seq> <talimat>` temiz bir proje açar, backend +
  sam_server'ı doğrular, Claude Code'u headless çalıştırır (`claude -p` ve skill), CLI
  çağrı günlüğünü (`.autoseg/cli_log.jsonl`) tutar, sonunda export'u GT ile puanlar.
- Talimatlar: "torpedo panosunu etiketle" (sınıf adı ve kare aralığı verilmeden), stres
  dizileri için aynısı, bir de iki nesneli senaryo (veri varsa).
- Toplam en fazla 2 koşu (az senaryo: torpedo + en fazla bir stres dizisi).
- Metrikler: WP3 metrikleri + CLI çağrı sayısı, görüntülenen görsel sayısı, süre, token,
  insan müdahalesi (hedef 0). Oracle ile yan yana rapor.

Kabul:
- torpedo: ort. IoU oracle'ın 0.02 altından kötü değil, koşularda insan müdahalesiz
- stres dizileri: oracle'ın 0.05 altından kötü değil
- başarısızlıklar sınıflanır (skill eksikliği / CLI eksikliği / motor / sinyal) ve ilgili
  WP'ye geri beslenir; ikinci tur deneme.

## WP8 · DINO fallback ve uzun diziler (M, koşullu)

WP3 ya da WP7, SAM'in kendiliğinden geri yakalayamadığı kayıplar gösterirse:

- `lost` issue'larında keyframe embedding'leriyle benzerlik araması (`ml/propagate.py`
  tepe tabanlı yol) → en iyi tepeden image maske → boyut + benzerlik kontrolü.
- Sonuç **aday keyframe** olarak issue'ya eklenir; CLI'da `track issues` adayı ve
  önizleme PNG'sini gösterir, ajan `track refine --preview` ile kabul eder. Otomatik kabul
  yok.
- Pencereleme (WP1'deki arayüz) ve çok nesne testleri.

Kabul: `exit`/`occl` dizilerinde aday bulma oranı ve ajanın müdahale sayısındaki düşüş
ölçülmüş.

## WP9 · Frontend (S–M, paralel, opsiyonel)

İnsanın ajanın işini izlemesi ve gerekirse devralması için: track zaman çizelgesinde
keyframe ve issue işaretleri, issue'ya tıklayınca kareye gitme, keyframe silme.
Mevcut `needs_review` görünümü yeniden kullanılır.

---

## Açık sorular

1. Ek etiketli diziler gelecek mi (çıkıp giren nesne, örtülme, iki nesne, sınıf 7)?
   Gelmezse WP3/WP7 sadece torpedo + türetilmiş stres dizileriyle değerlendirilir.
2. WP9 bu turda gerekli mi, yoksa ajan akışı bitince mi?
3. Ajan denemesinde hangi model (varsayılan: Claude Code'un varsayılan modeli)?
