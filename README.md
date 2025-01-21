# Verileri-Yuklemek-ve-Kullanilabilir-Hale-Getirmek

Orijinal veri setinde veriler 1280x720, fakat 1280x720 çözünürlüğü eğitim aşamasını çok daha masraflı hale getirecektir.

Bu yüzden daha düşük bir çözünürlük değeri olan 640x640 ı tercih ettim.

Bu çözünürlük değerleri training aşamasını daha ucuz ve hızlı hale getirirken biraz performansdan kaybedebiliriz.

Çünkü düşük çözünürlüklü data ile eğitilen model yüksek çözünürlüklü verilerde tahmin yaparken problem yaşayacaktır.

Burda doğru aralığı 640x640 olarak varsayabilirz. Çünkü 640x640 çözünürlük hem eğitimi çok masraflı bir hale getirmezken ayrıca yeterince detayda sağlayacaktır. 


