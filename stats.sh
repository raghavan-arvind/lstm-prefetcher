#!/bin/bash
echo "Submitting prefetching LSTM jobs"

GROUP='GRAD'
PROJECT='ARCHITECTURE'
DESCR='LSTM prefetcher stats collector'
GPU=false
EMAIL='pabstmatthew@cs.utexas.edu'

#ARRAY=('astar_163B' 'bwaves_1861B' 'bzip2_183B' 'cactusADM_734B' 'gcc_13B' 'GemsFDTD_109B' 'gobmk_135B' 'gromacs_1B' 'lbm_94B' 'leslie3d_1116B' 'libquantum_1210B' 'mcf_46B' 'milc_360B' 'omnetpp_340B' 'perlbench_53B' 'soplex_66B' 'sphinx3_2520B' 'wrf_1212B' 'xalancbmk_748B' 'zeusmp_600B')
ARRAY=('astar_163B' 'bwaves_1861B' 'GemsFDTD_109B' 'leslie3d_1116B' 'libquantum_1210B' 'milc_360B' 'omnetpp_340B' 'soplex_66B' 'sphinx3_2520B' 'xalancbmk_748B') 

ELEMENTS=${#ARRAY[@]}

input_dir="/scratch/cluster/zshi17/ChampSimulator/CRCRealOutput/0426-LLC-trace"
output_dir="/projects/coursework/2018-spring/cs395t-lin/lstm/500k-1r/"

for (( i=0; i<$ELEMENTS; i++))
do
    benchmark=${ARRAY[${i}]}
    trace_file="$input_dir/$benchmark"".txt"
    script_file="$input_dir/$benchmark""_stats.sh"
    stats_file="$output_dir/$benchmark""_d4.stats"
    condor_file="$input_dir/$benchmark""_stats.condor"

    if test -f $trace_file; then
        echo "Evaluating" $benchmark "from" $trace_file
        
        # create executable script
        echo "#!/bin/bash" > $script_file
        echo "export PATH=\"/opt/cuda-8.0/lib64:\$PATH\"" >> $script_file
        echo "export LD_LIBRARY_PATH=\"/opt/cuda-8.0/lib64:\$LD_LIBRARY_PATH\"" >> $script_file
        echo "export LD_LIBRARY_PATH=\"/u/matthewp/cuda/lib64:\$LD_LIBRARY_PATH\"" >> $script_file
        echo "python3 /u/matthewp/Downloads/decode_output.py $benchmark &> $stats_file" >> $script_file
        chmod +x $script_file

        # create condor file
        echo "+Group=\"$GROUP\"" > $condor_file
        echo "+Project=\"$PROJECT\"" >> $condor_file
        echo "+ProjectDescription=\"$DESCR\"" >> $condor_file
        echo "universe=vanilla" >> $condor_file
        echo "getenv=true" >> $condor_file
        echo "Rank=Memory" >> $condor_file
        echo "notification=Error" >> $condor_file
        echo "output=CONDOR.lstm.OUT" >> $condor_file
        echo "error=CONDOR.lstm.ERR" >> $condor_file
        echo "Log=CONDOR.lstm.LOG" >> $condor_file
        echo "notify_user=$EMAIL" >> $condor_file
        if [ "$GPU" = true ]; then
            echo "requirements=Cuda8 && TARGET.GPUSlot && CUDAGlobalMemoryMb >= 6144" >> $condor_file
            echo "request_GPUs=1" >> $condor_file
            echo "+GPUJob=true && NumJobStarts == 0" >> $condor_file
        fi
        echo "initialdir=$output_dir" >> $condor_file
        echo "executable=$script_file" >> $condor_file
        echo "queue" >> $condor_file

        # Submit the condor file
        /lusr/opt/condor/bin/condor_submit $condor_file
    fi
done

echo "Submitted all jobs"
