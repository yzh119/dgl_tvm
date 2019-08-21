echo "dataset, hidden-size, hidden-layers, tvm speed(ms), dgl speed(ms)"
for data in cora citeseer pubmed; do
  for D in 16 32 64 128; do
    for L in 1 5 10 15 20; do
      python tvm_gcn.py -data $data -d $D -l $L >/dev/null 2>/dev/null
      python dgl_gcn.py -data $data -d $D -l $L >/dev/null 2>/dev/null
      tvmdir="logs/tvm_num_hidden-"$L"_hidden_dim-"$D"_dataset-"$data
      dgldir="logs/dgl_num_hidden-"$L"_hidden_dim-"$D"_dataset-"$data
      echo $data, $D, $L, $(cat $tvmdir), $(cat $dgldir)
    done
  done
done
