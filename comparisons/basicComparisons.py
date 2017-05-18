# MatplotLib
import matplotlib.pyplot as plt
import numpy as np
import mpld3

prolangs = ('Python', 'C++', 'Java', 'Perl', 'Scala', 'Lisp')
y_pos = np.arange(len(prolangs))
fig = plt.figure()
performance = [10, 8, 6, 4, 2, 1]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, prolangs)
plt.ylabel('Usage')
plt.title('Programming language usage')
# mpld3.fig_to_html(plt.gcf())
mpld3.show(fig)

# Bokeh
from bokeh.charts import Bar, show
import pandas as pd

dictionary = {'languages': prolangs, 'values': performance}
df = pd.DataFrame(dictionary)

p = Bar(df, 'languages', values='values', title='Programming language usage')
show(p)

# d3
# df.to_csv('graph1-cumulative-reported-cases-all.txt',sep='\t',index=False,header=False)
'''<!DOCTYPE html>
<html lang="en">
<meta charset="utf-8">
<style>

.bar {
  fill: red;
}

.bar:hover {
  fill: green;
}

.axis {
  font: 10px sans-serif;
}

.axis path,
.axis line {
  fill: none;
  stroke: #000;
  shape-rendering: crispEdges;
}

.x.axis path {
  display: none;
}

</style>
<body>
<p> Test</p>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js"></script>
<script>

var margin = {top: 20, right: 20, bottom: 30, left: 40},
    width = 960 - margin.left - margin.right,
    height = 500 - margin.top - margin.bottom;

var x = d3.scale.ordinal()
    .rangeRoundBands([0, width], .1);

var y = d3.scale.linear()
    .range([height, 0]);

var xAxis = d3.svg.axis()
    .scale(x)
    .orient("bottom");

var yAxis = d3.svg.axis()
    .scale(y)
    .orient("left")
    .ticks(10, "");http://127.0.0.1:5000/pured3

var svg = d3.select("body").append("svg")
		.attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");



d3.tsv("/static/graph1-cumulative-reported-cases-all.txt", type, function(error, data) {
  if (error) throw error;

  x.domain(data.map(function(d) { return d.language; }));
  y.domain([0, d3.max(data, function(d) { return d.performance; })]);

  svg.append("g")
      .attr("class", "x axis")
      .attr("transform", "translate(0," + height + ")")
      .call(xAxis);

  svg.append("g")
      .attr("class", "y axis")
      .call(yAxis)
    .append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", 6)
      .attr("dy", ".71em")
      .style("text-anchor", "end")
      .text("Cases");


  svg.selectAll(".bar")
      .data(data)
    .enter().append("rect")
  		.attr({x: 400, y: 700})
  		.transition().duration(3000).ease("elastic").attr()
      .attr("class", "bar")
      .attr("x", function(d) { return x(d.language); })
      .attr("width", x.rangeBand())
      .attr("y", function(d) { return y(d.performance); })
      .attr("height", function(d) { return height - y(d.performance); });


});

function type(d) {
  d.performance = +d.performance;
  return d;
}
</script>
</body>
</html>'''