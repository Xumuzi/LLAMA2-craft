cd ..
cd src
python emb_exp.py --model_name Llama2-Chinese-7b-Chat --dataset_name lahate --method all_ct_single_emb --seed 999
python emb_exp.py --model_name Llama2-Chinese-7b-Chat --dataset_name lahate --method concat_ct_embs --seed 999
python emb_exp.py --model_name Llama2-Chinese-7b-Chat --dataset_name lahate --method avg_over_ct_embs --seed 999
python emb_exp.py --model_name Llama2-Chinese-7b-Chat --dataset_name lahate --method baseline --seed 999
python emb_exp.py --model_name Llama2-Chinese-7b-Chat --dataset_name toxicn --method all_ct_single_emb --seed 999
python emb_exp.py --model_name Llama2-Chinese-7b-Chat --dataset_name toxicn --method concat_ct_embs --seed 999
python emb_exp.py --model_name Llama2-Chinese-7b-Chat --dataset_name toxicn --method avg_over_ct_embs --seed 999
python emb_exp.py --model_name Llama2-Chinese-7b-Chat --dataset_name toxicn --method baseline --seed 999
