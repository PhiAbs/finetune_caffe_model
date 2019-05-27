import pandas as pd
import matplotlib as mpl
from matplotlib import *
mpl.use('Agg')
from matplotlib.pyplot import *
import sys


# name_prefix = "finetune_on_animal_gender_tennisball_soccerball_basketball"
name_prefix = "finetune_on_dog_cat_female_male_ball"
train_file = "./" + name_prefix + ".log.train"
test_file = "./" + name_prefix + ".log.test"

# increase tick label size
textsize = 20
matplotlib.rc('xtick', labelsize=textsize)
matplotlib.rc('ytick', labelsize=textsize)

if (len(sys.argv) > 1):
	print "create plot over " + sys.argv[1] +  " iterations"
	num_rows = int(sys.argv[1]) / 500  # for the validation (test) file, one row corresponds to 500 iterations. for the train file, one row corresponds to 20 iterations
	train_log = pd.read_csv(train_file, nrows = num_rows*25)
	test_log = pd.read_csv(test_file, nrows = num_rows+1)

else:
	print "create plot over all iterations"
        train_log = pd.read_csv(train_file)
        test_log = pd.read_csv(test_file)

_, ax1 = subplots(figsize=(15, 10))
# ax2 = ax1.twinx()
ax1.plot(train_log["NumIters"], train_log["loss_DL_PROJECT"], alpha=0.4)
ax1.plot(test_log["NumIters"], test_log["loss_DL_PROJECT"], 'g')
# ax2.plot(test_log["NumIters"], test_log["accuracy_DL_PROJECT"], 'r')
ax1.set_xlabel('iteration', fontsize=textsize)
ax1.set_ylabel('train loss (blue) / validation loss (green)', fontsize=textsize)
# ax2.set_ylabel('validation accuracy')
fig_name = "./" + name_prefix + ".png"
savefig(fig_name) #save image as png
