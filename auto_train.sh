: <<'END'
    *** Pretrain ***
    lr 0.001 / step 30 / gamma 0.2 / epochs 200

    *** QAT ***
    lr  / step  / gamma  / epochs 
END

# Learning rate 
python train.py --preload --load run/abpn-231104-22-09/weights/abpn_200.pth --qat --lr 0.01 --step 20 --gamma 0.5 --epochs 300
python train.py --preload --load run/abpn-231104-22-09/weights/abpn_200.pth --qat --lr 0.005 --step 20 --gamma 0.5 --epochs 300
python train.py --preload --load run/abpn-231104-22-09/weights/abpn_200.pth --qat --lr 0.001 --step 20 --gamma 0.5 --epochs 300

# Step
# python train.py --preload --load run/abpn-231104-22-09/weights/abpn_200.pth --qat --lr 0.001 --step 10 --gamma 0.5 --epochs 300
# python train.py --preload --load run/abpn-231104-22-09/weights/abpn_200.pth --qat --lr 0.001 --step 20 --gamma 0.5 --epochs 300
# python train.py --preload --load run/abpn-231104-22-09/weights/abpn_200.pth --qat --lr 0.001 --step 30 --gamma 0.5 --epochs 300
# python train.py --preload --load run/abpn-231104-22-09/weights/abpn_200.pth --qat --lr 0.001 --step 40 --gamma 0.5 --epochs 300
# python train.py --preload --load run/abpn-231104-22-09/weights/abpn_200.pth --qat --lr 0.001 --step 50 --gamma 0.5 --epochs 300

# Gamma
# python train.py --preload --load run/abpn-231104-22-09/weights/abpn_200.pth --qat --lr 0.001 --step 30 --gamma 0.2 --epochs 300
# python train.py --preload --load run/abpn-231104-22-09/weights/abpn_200.pth --qat --lr 0.001 --step 30 --gamma 0.5 --epochs 300
# python train.py --preload --load run/abpn-231104-22-09/weights/abpn_200.pth --qat --lr 0.001 --step 30 --gamma 0.8 --epochs 300



