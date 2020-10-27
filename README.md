<center>
<img src="https://www.schwarzwaelder-bote.de/media.media.c5d3a492-5f32-4bcc-83f3-27e779ad4d46.original1024.jpg" width="1000" align="center"/>
 <br><br>
 <h1>  <span style="color:green"> COVID-19 Open Research Dataset Challenge (CORD-19) </span>  </h1>
</center>
<h2> <span style="color:green"> Introduction </span>  </h2>
The aim of this notebook is to provide a robust algorithm that can help the medical research community to find useful information about COVID-19. 

<h3> <span style="color:green"> Approach </span> </h3> 
We designed a pipeline that consists of three parts: document retrieval, information extraction and creating html report for each query. 

<br><br><br>
![pipeline.png](https://user-images.githubusercontent.com/28005338/79320047-6ee29e00-7f09-11ea-887c-3b3f1cdfb09f.png)
<br><br><br>

The idea behind is to use the powerful BioBERT embeddings and use a traditional document retrieval technique (BM25+) to overcome some of its weaknesses.


**BM25** is the next generation of TFIDF and stands for “Best Match 25”. It is a ranking function used by search engines to rank matching documents according to their relevance to a given search query. It is based on the probabilistic retrieval framework.

**BioBERT** is a pre-trained biomedical language representation model for biomedical text mining.
<br><br><br>
<h3> <span style="color:green"> Difference with other approaches </span> </h3> 

We have seen that most approaches rely entirely on embedding techniques to find the answer or on document retrieval techniques to find the most relevant documents. We tried to use the best things of both approaches and combine them to overcome the weaknesses they have.

<h3> <span style="color:green"> Conclusions </span> </h3> 
By combining the document retrieval and the embedding comparison we can get results we couldn't get otherwise, besides reducing the computing time by comparing less documents.

<h3> <span style="color:green"> Next steps </span> </h3> 
We need to further process the output to obtain a more robust algorithm. Our main idea would be to implement a pretrained SNLI or QA model to improve the output given to the researcher in need for concrete and easy to read information. On the other hand we also will try to introduce topic selection in order to filter documents before information retrieval.
