for data in cora citeseer pubmed; do
  for D in 16 32 64 128; do
    for L in 1 5 10 15 20; do
      echo $data, $D, $L
      python tvm_gcn.py -data $data -d $D -l $L >/dev/null 2>/dev/null
      outdir="logs/tvm_num_hidden-"$L"_hidden_dim-"$D"_dataset-"$data
      echo "tvm speed: " $(cat $outdir) ms 
      outdir="logs/dgl_num_hidden-"$L"_hidden_dim-"$D"_dataset-"$data
      python dgl_gcn.py -data $data -d $D -l $L >/dev/null 2>/dev/null
      echo "dgl speed: " $(cat $outdir) ms
    done
  done
done
