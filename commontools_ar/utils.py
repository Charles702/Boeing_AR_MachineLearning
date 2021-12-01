from pkg_resources import resource_stream

def read_config():
  f = resource_stream('commontools_ar', 'resources/train_loss_detail_pdepoch50.csv')
  
  print(f.readline())
  print(__file__)