function createTOC(){
	console.log("createTOC started");
    var toc = "";
    var level = 0;
    var levels = {}
    $("#toc").html("");

    $(":header").each(function(i){
		console.log("LOOP: ", this, this.id, this.hasAttribute("data-skip-toc"));
        if (this.id =="tocheading" || this.hasAttribute("data-skip-toc")) {return;}
        
	    var titleText = this.innerHTML;
	    var openLevel = this.tagName[1];

	    if (levels[openLevel]){
		    levels[openLevel] += 1;
	    } else{
		    levels[openLevel] = 1;
	    }

	    if (openLevel > level) {
		    toc += (new Array(openLevel - level + 1)).join("<ul class=\"toc\">");
	    } else if (openLevel < level) {
		    toc += (new Array(level - openLevel + 1)).join("</ul>");
		    for (i=level;i>openLevel;i--){levels[i]=0;}
	    }

	    level = parseInt(openLevel);


	    if (this.id==""){this.id = this.innerHTML.replace(/ /g,"-")}
	    var anchor = this.id;
        
	    toc += "<li><a href=\"#" + encodeURIComponent(anchor) + "\">"
		+ romanize(levels[openLevel].toString()) + ". " + titleText
		+ "</a></li>";
        
	});

    
    if (level) {
	    toc += (new Array(level + 1)).join("</ul>");
    }

 
	$("#toc").append(toc);
	console.log("createTOC finished");
	console.log("**********************************************************************");
};

// Executes the createToc function
setTimeout(function(){createTOC();},100);

// Rebuild to TOC every minute
setInterval(function(){createTOC();},60000);