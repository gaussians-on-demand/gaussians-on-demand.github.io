document.addEventListener('DOMContentLoaded', domReady);

        function domReady() {
            new Dics({
                container: document.querySelectorAll('.b-dics')[0],
                hideTexts: false,
                textPosition: "bottom"

            });
            new Dics({
                container: document.querySelectorAll('.b-dics')[1],
                hideTexts: false,
                textPosition: "bottom"

            });
        }

        function objectSceneEvent(idx) {
            let dics = document.querySelectorAll('.b-dics')[0]
            let sections = dics.getElementsByClassName('b-dics__section')
            let imagesLength = 8
            for (let i = 0; i < imagesLength; i++) {
                let image = sections[i].getElementsByClassName('b-dics__image-container')[0].getElementsByClassName('b-dics__image')[0]
                switch (idx) {
                    case 0:
                        image.src = './static/images/bicycle';
                        break;
                    case 1:
                        image.src = './static/images/bonsai';
                        break;
                    case 2:
                        image.src = './static/images/counter';
                        break;
                    case 3:
                        image.src = './static/images/flowers';
                        break;
                    case 4:
                        image.src = './static/images/garden';
                        break;
                    case 5:
                        image.src = './static/images/kitchen';
                        break;
                    case 6:
                        image.src = './static/images/room';
                        break;
                    case 7:
                        image.src = './static/images/stump';
                        break;    
                    case 8:
                        image.src = './static/images/treehill';
                        break;    
                    case 9:
                        image.src = './static/images/truck';
                        break;    
                    case 10:
                        image.src = './static/images/train';
                        break;    
                    case 11:
                        image.src = './static/images/drjohnson';
                        break;    
                    case 12:
                        image.src = './static/images/playroom';
                        break;    
                }
                switch (i) {
                    case 0:
                        image.src = image.src + '/L0.jpeg';
                        break;
                    case 1:
                        image.src = image.src + '/L1.jpeg';
                        break;
                    case 2:
                        image.src = image.src + '/L2.jpeg';
                        break;
                    case 3:
                        image.src = image.src + '/L3.jpeg';
                        break;
                    case 4:
                        image.src = image.src + '/L4.jpeg';
                        break;
                    case 5:
                        image.src = image.src + '/L5.jpeg';
                        break;
                    case 6:
                        image.src = image.src + '/L6.jpeg';
                        break;
                    case 7:
                        image.src = image.src + '/L7.jpeg';
                        break;
                }
            }

            let scene_list = document.getElementById("object-scale-recon").children;
            //console.log(scene_list)
            for (let i = 0; i < scene_list.length; i++) {
                console.log("OK")
                if (idx == i) {
                    scene_list[i].children[0].className = "nav-link active"
                }
                else {
                    scene_list[i].children[0].className = "nav-link"
                }
            }
        }