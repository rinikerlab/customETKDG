# customETKDG

## Docker 

### Build
```
#if build failed with error code 137, increase the RAM allocated to Docker.
docker build -t custom_etkdg .
```

### Run
```
#as interactive bash
docker run -it --entrypoint /bin/bash custom_etkdg:latest

#as jupyter session, to run the demo notebook in `examples` folder
docker run -p 13579:13579 custom_etkdg
```
