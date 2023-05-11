shot=45
for i in 0 1 2 3 4
do
    python evaluate_on_gpt3_api_few_shot.py --prompt_path datasets/downstream/STS/STS12-en-test/sts2012-train/${shot}shot/demons/${shot}shot_random${i}.txt \
    --dump_path datasets/downstream/STS/STS12-en-test/sts2012-train/${shot}shot/no_reasons/random${i}devALL \
    --task_description_id 3 \
    --example_template_id 1 \
    --testdata_path datasets/downstream/STS/STS12-en-test/sts2012-train/dev.txt
done