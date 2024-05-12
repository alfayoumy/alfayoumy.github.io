(function(document) {

    var topbtn = document.getElementById("topbtn");
    var footer = document.getElementsByTagName('footer')

    function isScrolledIntoView(el) {
        var elemTop = el.getBoundingClientRect().top;
        var elemBottom = el.getBoundingClientRect().bottom;
        return (elemTop >= 0) && (elemBottom <= window.innerHeight);
    }

    window.onscroll = function() {scrollFunction()};

    function scrollFunction() {
        if ((document.body.scrollTop > 30 || document.documentElement.scrollTop > 30) && !(isScrolledIntoView(footer[0])) ) {
            topbtn.style.display = "block";
        } else {
            topbtn.style.display = "none";
        }
    }

    // Scroll to top
    function topFunction() {
        document.body.scrollTop = 0;
        document.documentElement.scrollTop = 0;
    }

    var metas = document.getElementsByTagName('meta'),
        changeViewportContent = function(content) {
            for (var i = 0; i < metas.length; i++) {
                if (metas[i].name == "viewport") {
                    metas[i].content = content;
                }
            }
        },
        initialize = function() {
            changeViewportContent("width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no");
        },
        gestureStart = function() {
            changeViewportContent("width=device-width, minimum-scale=0.25, maximum-scale=1.6");
        },
        gestureEnd = function() {
            initialize();
        };


    if (/Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)) {
        initialize();
        document.addEventListener("touchstart", gestureStart, false);
        document.addEventListener("touchend", gestureEnd, false);
    }
})(document);

