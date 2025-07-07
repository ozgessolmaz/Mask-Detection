Transfer Learning ile Maske Tespiti Raporu 


-Ödevi yapanlar; 
Özge Solmaz 
Nisasu Bozkurt 
Rukiye Uçar 


-Bu projede, MobileNetV2 tabanlı transfer learning yöntemiyle insan yüzü görsellerinde maske takılı olup olmadığını sınıflandıran br model geliştirilmiştir.  
Veriler Kaggle'dan alınan 'Face Mask Detection Dataset' üzerinden alınmış, %80 eğitim, %20 doğrulama şeklinde ayrılmıştır. 
Görseller 224x224 boyutuna getirilmiş ve normalize edilmiştir (0i1 aralığına). 
 
Model Mimarisi 
-	Base Model: MobileNetV2 (ImageNet ağırlıklarıyla) 
-	GlobalAveragePooling2D 
-	Dense(128, relu) 
-	Dropout(0.3) 
-	Dense(1, sigmoid) 
Loss fonksiyonu: binary_crossentropy 
Metric: accuracy 
 
Eğitim Süreci 
Model 10 epoch boyunca eğitilmiş, erken durdurma (patience=3) uygulanmıştır. Eğitim süreci sonunda en iyi doğrulama performansı olan ağırlıklar restore edilmiştir. 
Aşağıda eğitim sırasında elde edilen doğruluk ve kayıp grafiklerini bulabilirsiniz: 

 ![image](https://github.com/user-attachments/assets/f6398bf1-c4c9-4a3e-b69f-29a13c799aa7)

Confusion Matris’x & Metrikler 
 ![image](https://github.com/user-attachments/assets/329e338e-6c24-4970-babb-9da217586fdc)
 
Confusion Matrix: 
![image](https://github.com/user-attachments/assets/5ace0a46-670d-45b0-95e7-3e713bc233fe)

Classification Report: 

![image](https://github.com/user-attachments/assets/711f5a68-890a-4cc4-a663-0c3cbed11759)
 
 
Örnek Görseller Üzerinde Tahminler 
Modelin doğrulama setinden seçilen 5 rastgele görsel üzerindeki tahminleri aşağıda görebilirsiniz: 
 ![image](https://github.com/user-attachments/assets/a4edbed2-1f07-4889-91ac-1a84462921e0)


Karşılaşılan Zorluklar ve Çözümler 
-Eğitim sırasında val_loss dalgalanması görüldü, bu nedenle early stopping uygulandı. 
 
