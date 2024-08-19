# CNFformer

To Run experiments, download all the files from the respective folder into a single local directory. Some files are common across different experiments. However, they have been provided again in the respective folder to avoid confusion. 


To download models, please refer to the gdrive link in the folder MODELS. Three models are provided, baseline, fine-tuned-epoch50, and fine-tuned-epoch 65.
To download data for comparison experiments, refer to Minimised Circuits folder. The rest files in Data directory are for running structured problems experiment.


All commands are to be run on the command line. 
A GPU with 48GB space was used to run all the experiments. For quick inference, run on GPU. Code can be run on cpu too but inference time will be slow. Script will automatically select GPU if available.
Detailed instructions to run each experiment in a README document in the respective folders.



Notes on results:
Please note that there would be some variablity in the results, as they are averaged across several runs. However, the overall trends and conclusions (hypotheses) remain consistent. This variability is a known aspect of the methods employed and does not affect the validity of the comparisons or the conclusions drawn from the experiments. 
