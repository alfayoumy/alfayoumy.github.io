(function(document) {
    var metas = document.getElementsByTagName('meta'),
        changeViewportContent = function(content) {
            for (var i = 0; i < metas.length; i++) {
                if (metas[i].name == "viewport") {
                    metas[i].content = content;
                }
            }
        },
        initialize = function() {
            // Check if the device is a desktop or mobile
            if (/Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)) {
                // For mobile devices
                changeViewportContent("width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no");
            } else {
                // For desktop devices
                changeViewportContent("width=device-width, initial-scale=1.5");
            }
        },
        gestureStart = function() {
            changeViewportContent("width=device-width, minimum-scale=0.25, maximum-scale=1.6");
        },
        gestureEnd = function() {
            initialize();
        };


    initialize();
    if (/Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)) {
        document.addEventListener("touchstart", gestureStart, false);
        document.addEventListener("touchend", gestureEnd, false);
    }
    
})(document);