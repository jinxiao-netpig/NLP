# PETER (PErsonalized Transformer for Explainable Recommendation)

## Paper
- Lei Li, Yongfeng Zhang, Li Chen. [Personalized Transformer for Explainable Recommendation](https://lileipisces.github.io/files/ACL21-PETER-paper.pdf). ACL'21.

## Usage
Below are examples of how to run PETER (with and without the key feature).
```
python -u main.py \
--data_path ../TripAdvisor/reviews.pickle \
--index_dir ../TripAdvisor/1/ \
--cuda \
--checkpoint ./tripadvisorf/ \
--peter_mask \
--use_feature >> tripadvisorf.log

python -u main.py \
--data_path ../TripAdvisor/reviews.pickle \
--index_dir ../TripAdvisor/1/ \
--cuda \
--checkpoint ./tripadvisor/ \
--peter_mask >> tripadvisor.log
```

## Citation
```
@inproceedings{ACL21-PETER,
	title={Personalized Transformer for Explainable Recommendation},
	author={Li, Lei and Zhang, Yongfeng and Chen, Li},
	booktitle={ACL},
	year={2021}
}
```
