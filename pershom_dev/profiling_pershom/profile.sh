export LD_LIBRARY_PATH="/usr/local/lib:/scratch2/chofer/opt/anaconda3/envs/pyt_gh/lib/:/scratch2/chofer/opt/anaconda3/envs/pyt_gh/lib/python3.6/site-packages/torch/lib"

rm -f profile

echo "Compiling ..."

make

make clean


echo "Compiled! Now executing ... "
./profile 


