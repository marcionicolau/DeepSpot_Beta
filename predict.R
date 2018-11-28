library(tensorflow)


predict_leafdisease = function(img_path='./test0.jpg', top=2) {
  
  labels = readLines('model0/saved_model.txt')
  image_data = tf$gfile$FastGFile(img_path, 'rb')$read()
  
  with(tf$gfile$FastGFile('model0/saved_model.pb', 'rb') %as% f, {
    graph_def = tf$GraphDef()
    graph_def$ParseFromString(f$read())
    tf$import_graph_def(graph_def, name='')
  })
  
  
  with(tf$Session() %as% sess, {
    softmax_tensor = sess$graph$get_tensor_by_name('final_result:0')
    predictions = sess$run(softmax_tensor, dict('DecodeJpeg/contents:0'= image_data))
    top_k = order(predictions,decreasing = T)
    data.frame(label_class=labels[top_k[seq_len(top)]], scores=predictions[top_k[seq_len(top)]])
  })
  
}


