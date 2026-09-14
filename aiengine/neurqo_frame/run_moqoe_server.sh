cd /code/neurdb-dev/aiengine/neurqo_frame
export PYTHONPATH="$(pwd)/src:$(pwd)/src/expert_pool/hint_plan_sel_expert:$(pwd)/src/expert_pool/join_order_expert:$PYTHONPATH"

mkdir -p logs
chmod 755 logs

if [ "$1" == "sudo" ]; then
    sudo -E /home/neurdb/.conda/envs/moqoe/bin/python run.py
else
    /home/neurdb/.conda/envs/moqoe/bin/python run.py
fi
