export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "Rank 0 node is at $MASTER_ADDR"
MASTER_PORT="$(( $SLURM_JOB_ID % 16384 + 49152 ))"
export MASTER_PORT;
echo "Will use port $MASTER_PORT for c10d communication"
export WORLD_SIZE="$(($SLURM_NNODES * $SLURM_GPUS_ON_NODE))"
echo "WORLD_SIZE = $WORLD_SIZE"
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=bond0
export TORCH_NCCL_BLOCKING_WAIT=1
torchrun  --nnodes="$SLURM_JOB_NUM_NODES" \
          --nproc_per_node="$SLURM_GPUS_ON_NODE" \
          --rdzv_id="$SLURM_JOB_ID" \
          --rdzv_backend=c10d \
          --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
	  pretrain.py \
	   --data_dir=/scratch/ssd004/scratch/pmillana/CANADA_1.5M/data/ \
	   --log-wandb \
	   --run-name="test_1" \
	   --run-id="fiesta"
