# time-series-adversarials
Adversarial examples on time series data

## Code: <br>
*/models/* : Includes the architectures that we implemented. For shapelet based model please check */models/shapelet.py <br>
***main.py*** includes the main code to run the experiments. <br>

Results will be saved to *./results/DATASET_NAME/ARCHITECTURE_NAME/* by default. Checkpoints can be found there. <br>

To run an experiment, please use the following command: <br>
***python main.py --device=cpu --model=ShapeletNet --dataset=UCR_Wine*** <br>
To evaluate your model against adversarial attack, run : <br>
***python main.py --device=cpu --model=ShapeletNet --dataset=UCR_Wine --resume --adversarial-eval*** <br>

You can use any dataset name in timeseriesclassification.com by specifying UCR_Name. <br>

Current Models: resnet, MLP, FCN and ShapeletNet <br>
## Experimental
Add *--distance-loss* to the command-line args in order to use the loss that encourages shapelets to be distant from each other. Works only for shapeletnet. <br>
