[comment]: <> ([![Contributors][contributors-shield]][contributors-url])

[comment]: <> ([![Forks][forks-shield]][forks-url])

[comment]: <> ([![Stargazers][stars-shield]][stars-url])

[comment]: <> ([![Issues][issues-shield]][issues-url])

[comment]: <> ([![MIT License][license-shield]][license-url])

[comment]: <> ([![LinkedIn][linkedin-shield]][linkedin-url])


<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/Gregoire-z/Projet_IoT">

![logo](others/logo.png)

<h3 align="center">Project IoTracker</h3>

  <p align="center">
   IoT Operator Information System Project - Visualization 
    <br />
    <a href="https://iotracker.fr/"><strong>View Demo »</strong></a>
    <br />
    <br />
  </p>




<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul> </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#to-run-docker">To Run Docker</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

![About the project](./others/about.png)


## Getting Started

### Installation
#### Docker


1. Install [docker](https://www.docker.com/get-started), and preferably also install docker-desktop.
2. Now you need to install docker compose, to do so run the following command to dl:
    ```bash
    sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    ```
   
3. Now we need to apply executable permissions to the binary to allow us to run compose
    ```bash
    sudo chmod +x /usr/local/bin/docker-compose
   ```

### To Run Docker
Type in the terminal (where the docker-compose.yml is), to run the project :
 ```bash
 docker-compose up -d
 ```

This will launch 4 containers : php-Apache, Python, HaProxy and Postgresql. 


![Image of Docker](others/terminalDockerComposeUp.png)

Thanks to docker, you do not need to install any software packages to set up a local Web servers like XAMPP or MAMP. 
Everything is managed by docker, because it turns on an apache web server.


When every containers is running like this: 

![Image of Terminal](others/imageDocker.png)

**Go to :**
[http://localhost/log/login.php](http://localhost/log/login.php)

![Image of login Page](others/login.png)


### To shutdown docker
To shutdown all the containers 

```bash
docker-compose down
```

### To build all the images of your project docker
To rebuild all your images Docker 
```bash
docker-compose build
```



### To create the volume docker

```bash
docker volume create --name=project_iot
```


### To delete all the volume docker

```bash
docker volume prune 
```

### To delete all the dangling images in docker

```bash
docker image prune  
```



<!-- CONTACT -->
## Contact

* Gregoire - [Github](https://github.com/Gregoire-z)  (PHP Apache)
* Kenan - [Github](https://github.com/kenanGonnot)  (Docker)
* Ezechiel - [Github](https://github.com/Rellfix)  (PHP Apache)
* Carla - [Github](https://github.com/Gregoire-z)  (AWS)
* Quentin - [Github](https://github.com/Quentin932) (PHP Apache)
* Lorenzo - [Github](https://github.com/Lorenzo089) (Objenious)
* Jules - [Github](https://github.com/Terrent3) (Objenious)

Tuteur : Philippe Cola - [@esme.fr](philippe.cola@esme.fr)

Project link: [https://iotracker.fr](https://iotracker.fr)






<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/kenan-gonnot-4037a6197/
[product-screenshot]: images/screenshot.png


