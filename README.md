# customETKDG

## Docker 

### Build
```
docker build -t custom_etkdg .
```

### Run
```
#as interactive bash
docker run -it --entrypoint /bin/bash custom_etkdg:latest

#as jupyter session
docker run -p 13579:13579 custom_etkdg
```
