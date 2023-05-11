python evaluate_on_gpt3_api_knn_few_shot.py --testdata_path datasets/downstream/STS/STSBenchmark/test-ALL.txt \
--prompt_path datasets/downstream/STS/STSBenchmark/KNN500/25-deberta-lr9e-6-bs16-manual-explanation/test/demons \
--task_description_id 3 \
--example_template_id 3 \
--dump_path datasets/downstream/STS/STSBenchmark/KNN500/25-deberta-lr9e-6-bs16-manual-explanation/test/reasons/testALL \
--max_tokens 100