import matplotlib.pyplot as plt
import numpy as np
import mpld3
from mpld3 import plugins


class Update(plugins.PluginBase):  # inherit from PluginBase
    """Hello World plugin"""

    JAVASCRIPT = """
    mpld3.register_plugin("update", Update);
    Update.prototype = Object.create(mpld3.Plugin.prototype);
    Update.prototype.constructor = Update;
    function Update(fig, props){
        mpld3.Plugin.call(this, fig, props);
    };

    Update.prototype.draw = function(){
        this.fig.canvas.append("text")
            .text("hello world")
            .style("font-size", 72)
            .style("opacity", 0.3)
            .style("text-anchor", "middle")
            .attr("x", this.fig.width / 2)
            .attr("y", this.fig.height / 2)
    }
    """

    def __init__(self):
        self.dict_ = {"type": "update"}


fig, ax = plt.subplots()
plugins.connect(fig, Update())
mpld3.show()
