# Learning rate 
# python train.py --preload --lr 0.01 --step 20 --gamma 0.5 --epochs 300
# python train.py --preload --lr 0.005 --step 20 --gamma 0.5 --epochs 300
# python train.py --preload --lr 0.001 --step 20 --gamma 0.5 --epochs 300


# # Step
# # Epoch, Learning rate 결정되면 코드 돌리기
python train.py --preload --lr 0.001 --step 10 --gamma 0.5 --epochs 300
python train.py --preload --lr 0.001 --step 20 --gamma 0.5 --epochs 300
python train.py --preload --lr 0.001 --step 30 --gamma 0.5 --epochs 300
python train.py --preload --lr 0.001 --step 40 --gamma 0.5 --epochs 300
python train.py --preload --lr 0.001 --step 50 --gamma 0.5 --epochs 300

# # Gamma
# # Epoch, lr, step 결정되면 코드 돌리기
# python train.py --preload --lr  --step --gamma 0.2 --epochs 
# python train.py --preload --lr  --step --gamma 0.5 --epochs 
# python train.py --preload --lr  --step --gamma 0.8 --epochs 



