# Fraud Detection Menggunakan Graph Neural Networks
Mendeteksi kecurangan dalam grafik transaksi menggunakan berbagai model dan dataset siap pakai.

## Kelompok 10

- Abdullah Asy-Syifawi (2408107010042)
- Ahmad Daniel Chalid (2408107010061)
- Muhammad Irfan Qadafi (2408107010054)
- Rahiqul Munadhil (2408107010049)

Datasets:
[Elliptic](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)

## Instalasi
First install the requirements.
```bash
pip install -r requirements.txt
```

## Output
run command
```bash
python visualize.py --config configs/elliptic_gat.yaml --step 30 --weights_file weigths/elliptic_gat.pt
```
![A sample predictions visualization](visualizations/elliptic_gat/30.png)
