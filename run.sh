#pip install tslearn
export GIT_PYTHON_REFRESH=quiet
for MODEL in resnet FCN MLP
do
    for DS in ECG200 Strawberry GunPoint OSULeaf Earthquake ShapeletSim Wine ElectricDevices
    do
        python3 main.py --base-dir=/om2/user/merty/dummyresult/t/ --model=$MODEL --dataset=UCR_$DS
        python3 main.py --base-dir=/om2/user/merty/dummyresult/t/ --model=$MODEL --dataset=UCR_$DS --resume --adversarial-eval
    done
done
