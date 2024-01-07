for FILE in *; do
	echo ${FILE}
	sbatch ${FILE}
	sleep 1
done
