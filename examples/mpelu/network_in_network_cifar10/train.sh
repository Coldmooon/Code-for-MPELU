xelu="mpelu"
init="gauss"
lr="0.1"
directory="trained_models"
dataset="cifar10"
unified="y"
# solver="solver_$xelu.prototxt"

#for j in 1 2; do
for i in 1, 2, 3, 4, 5;
do
# generating solver.prototxt
# -----------------------------
# train_val_dprelu_msra_thresh-15-part1-train
# train_val_dprelu_msra_thresh-15-part2-test
# if [ $i = 10 ];
# then
# 	echo "Switch activation function from ${xelu} to \c"
# 	xelu="prelu"
# 	echo "${xelu}"
# fi 

if [ "${unified}" = "y" ];
then
	echo "net: \"models/network_in_network/nin_${xelu}_${init}.prototxt\"" > solver.prototxt
else
	echo "train_net: \"models/network_in_network/train_val_${xelu}_${init}.prototxt\"" > solver.prototxt
	echo "test_net: \"models/network_in_network/train_val_${xelu}_${init}.prototxt\"" >> solver.prototxt
fi
echo "test_iter: 100" >> solver.prototxt
echo "test_interval: 1000" >> solver.prototxt
echo "test_initialization: false" >> solver.prototx
echo "base_lr: ${lr}" >> solver.prototxt
echo "momentum: 0.9" >> solver.prototxt
echo "weight_decay: 0.0001" >> solver.prototxt
echo "lr_policy: \"multistep\"" >> solver.prototxt
echo "gamma: 0.1" >> solver.prototxt
echo "stepvalue: 100000" >> solver.prototxt
echo "display: 100" >> solver.prototxt
echo "max_iter: 120000" >> solver.prototxt
echo "snapshot: 10000" >> solver.prototxt
echo "snapshot_prefix: \"${xelu}_${init}\"" >> solver.prototxt
echo "solver_mode: GPU" >> solver.prototxt
# ------------------------------
echo 
cat solver.prototxt
echo 
echo "Training"

	echo "training network ${i}"
	time build/tools/caffe train --solver=solver.prototxt -gpu 0 \
	                        > nin_${xelu}_${init}_${lr}_${i}.txt 2>&1
	now=$(date +"%Y%m%d_%H_%M")
	mkdir $directory
	mv nin_${xelu}_${init}_${lr}_${i}.txt ${directory}/nin_${xelu}_${init}_${lr}_${now}.txt
	mv ${xelu}_${init}_* ${directory}/
	mv $directory nin_${xelu}_${init}_${lr}_${dataset}_${now}
	echo "network ${i} done!";
	echo 
done
#done
echo "Training is complete!"
