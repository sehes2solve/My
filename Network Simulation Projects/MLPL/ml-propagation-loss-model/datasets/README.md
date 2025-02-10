# MLPL Datasets

This folder contains the datasets to train the ML models of the path loss and fast-fading.



## Dataset Directory Structure

```text
datasets/
  {MY_DATASET}/
    dataset/
      propagation-loss-dataset.csv
    ml-model/
      {ML_MODEL_INPUT}/
        {ML_TRAINING_ALGORITHM}/
          fast-fading-ecdf.csv
          fast-fading-ecdf.pickle
          path-loss.pickle
    results/
```

## Dataset Format


### Category: Node Positions

| CSV Header | CSV Value | Supported Models |
| ---------- | --------- | ---------------- |
| distance_m | Distance between the Tx and Rx nodes in meters | D-MLPL |
| x_tx, y_tx, z_tx, x_rx, y_rx, z_rx | Tx and Rx nodes positions in meters | P-MLPL |

### Category: Propagation Loss

| CSV Header | CSV Value | Supported Models |
| ---------- | --------- | ---------------- |
| loss_db | Total propagation loss in dB | All |
| rx_power_dbm | Rx power in dBm | All |
| snr_db, noise_dbm | Signal-to-Noise Ratio (SNR) in dB and noise in dBm | All |

### Category: Optional Data

| CSV Header | CSV Value | Supported Models |
| ---------- | --------- | ---------------- |
| throughput_kbps | Throughput of the frame in kbit/s | All |
