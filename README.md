
checkpoints를 다운받아서

checkpoints/EmbeddingModel_MERT_768-epoch=0073-val_loss=0.1058-val_acc=0.9585-val_f1=0.9366-val_precision=0.9936-val_recall=0.8857.ckpt

checkpoints/step=007000-val_loss=0.1831-val_acc=0.9278.ckpt

이렇게 배치해두고 

inference.py

코드를 돌려서 음원 하나정도 되는지 확인해보고 쓰면됩니다

이 Readme는 내용 인지후에 inference.py 돌려서 돌아가면 프로젝트용으로 완전히 수정해주시면 됩니다

P.S. 위 체크포인트는 FST아니고 그냥 Segment Transformer라 이 레포지토리에서 수정 작업 들어갈겁니다.. 이거는 체크포인트 바꾸고 인자하나 바꾸면 끝이라 금방 바꿔져서 들어가요


