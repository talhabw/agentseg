# SAM3 video evaluation results

| Run | Model | Prompt | Mean IoU | Hit rate | First loss | Direction | Corrections |
|---|---|---|---:|---:|---:|---|---:|
| A_text_sign_board | sam3.1 | semantic | 0.9559 | 1.0 | None | forward |  |
| A_text_white_board | sam3.1 | semantic | 0.9562 | 1.0 | None | forward |  |
| A_text_white_board_pf263_both | sam3.1 | semantic | 0.9575 | 1.0 | None | both |  |
| B_torpedo_board | sam3.1 | semantic | 0.0056 | 0.0057 | 0 | forward |  |
| B_white_board | sam3.1 | semantic | 0.9559 | 1.0 | None | forward |  |
| C_exemplar_only | sam3.1 | semantic | 0.0055 | 0.0057 | 0 | forward |  |
| D_point1_sam3 | sam3 | points | 0.9605 | 1.0 | None | forward |  |
| D_point1_sam31 | sam3.1 | points | 0.5291 | 0.7624 | 0 | forward |  |
| D_point7_sam31 | sam3.1 | points | 0.9582 | 1.0 | None | forward |  |
| E_mask_sam3 | sam3 | mask | 0.9626 | 1.0 | None | forward |  |
| F_point1_sam3_pf441_both | sam3 | points | 0.4854 | 0.4962 | 0 | both | 10 |
| F_point1_sam3_pf441_both_corrmask | sam3 | points | 0.8447 | 0.9829 | None | both | 1 |
| F_point1_sam3_pf441_both_corrmask_loss08 | sam3 | points | 0.9618 | 1.0 | None | both | 6 |

`E_mask_sam3` (mask prompt via image head then `add_mask`) is the chosen method.
