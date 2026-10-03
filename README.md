DENOISING DIFFUSION PROBABILISTIC MODEL TRAINED ON MNIST HANDWRITTEN DIGITS
| Parameter | Value |
| --- | --- |
| **Optimizer** | AdamW |
| **Weight Decay** | $1 \times 10^{-2}$ (`1e-2`) |
| **Learning Rate** | $2 \times 10^{-4}$ (`2e-4`) |
| **Loss Function** | MSE (Mean Squared Error) |
| **Image Dimensions** | $1 \times 28 \times 28$ |
| **Batch Size** | 128 |
| **Timesteps embedding** | Sinusoidal |
| **Timesteps embedding dimensions** | 128 |
| **Epochs** | 10 |
| **Final Loss** | 0.02287 |


RUN TRAIN.PY THEN SAMPLE.PY
<h1>SAMPLING RESULT</h1>
<table>
  <tr>
    <td align="center"><img src="https://github.com/user-attachments/assets/f6bf57ca-6138-4ea8-a8f1-6e972a7e493a" width="100%" /></td>
    <td align="center"><img src="https://github.com/user-attachments/assets/49b10070-9dba-4da8-86fe-e649a785682e" width="100%" /></td>
  </tr>
  <tr>
    <td align="center"><img src="https://github.com/user-attachments/assets/3f14abec-ea58-4eff-b4fd-8fd110d14fa3" width="100%" /></td>
    <td align="center"><img src="https://github.com/user-attachments/assets/abe8a7c8-291e-48f3-b62f-c803ff6a909c" width="100%" /></td>
  </tr>
</table>
