# Docker

Requirements:
- docker
- docker-compose

Files:
- Dockerfile
- docker-compose.yml 

Volumes: The *Training* and *Application* folders will be mounted into
the Docker container. See docker-compose.yml.

Inside this folder, create a docker image:

``` shellsession
	docker build -t autoqc-dev .
```

Run docker compose:

``` shellsession
	docker compose up
```

Enter container:

``` shellsession
	docker exec -it docker-autoqc-1 bash
```

Go into Training or Application folders:

``` shellsession
	cd /root/Training
```

``` shellsession
	cd /root/Application
```

Run the scripts.


