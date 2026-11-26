"""v0.4 dataset sanity check without fitting a model."""
import argparse
from oibc_v3_common import get_schema,find_data,read_v3,train_mask
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default=None);a=p.parse_args()
    path=find_data(a.data)
    print('입력 파일:',path)
    print('전체 parquet 컬럼:',get_schema(path))
    train,dev,_,cats,nums,_=read_v3(path)
    use=train.loc[train_mask(train)]
    print('학습 대상:',len(use),'행, 발전소',use.site_id.nunique(),'개')
    print('검증 대상:',len(dev),'행, 발전소',dev.site_id.nunique(),'개')
    print('선택 특징:',cats+nums)
