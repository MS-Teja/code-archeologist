<template>
    <div class="codebase-heatmap container">
        <h2>Codebase Heatmap</h2>
        <svg ref="heatmap"></svg>
        <div class="legend">
            <span><span class="color-box low"></span> Low Changes</span>
            <span><span class="color-box medium"></span> Medium Changes</span>
            <span><span class="color-box high"></span> High Changes</span>
        </div>
        <i>Hover over the heatmap for more details</i>
    </div>
</template>

<script>
    import { ref, onMounted } from 'vue';
    import axios from 'axios';
    import * as d3 from 'd3';
    import { store } from '../store/analysis';

    export default {
        name: 'CodebaseHeatmap',
        data() {
            return {
                store,
            };
        },
        setup() {
            const heatmap = ref(null);
            const fetchData = async () => {
                try {
                    const response = await axios.get('/api/codebase-heatmap', {
                    withCredentials: true,
                    headers: { 'X-Analysis-ID': store.analysisId }
                    });

                    if (response.data.status === 'success' && response.data.data) {
                        const fileChanges = response.data.data;
                        // Ensure fileChanges is an object
                        if (typeof fileChanges === 'object' && Object.keys(fileChanges).length > 0) {
                            const data = buildHierarchy(fileChanges);
                            renderHeatmap(data);
                        } else {
                            console.warn('No file changes data available');
                        }
                    }
                }
                catch (error) {
                    console.error('Error fetching codebase heatmap data:', error);
                }
            };
            const buildHierarchy = (fileChanges) => {
                const root = { name: 'root', children: [] };

                Object.entries(fileChanges).forEach(([path, count]) => {
                    const parts = path.split('/');
                    let current = root;

                    parts.forEach((part, index) => {
                        let child = current.children.find(c => c.name === part);
                        if (!child) {
                            child = {
                                name: part,
                                children: [],
                                value: index === parts.length - 1 ? count : undefined
                            };
                            current.children.push(child);
                        }
                        current = child;
                    });
                });

                return root;
            };
            const renderHeatmap = (data) => {
                const width = 800;
                const height = 600;
                const svg = d3.select(heatmap.value)
                    .attr('width', width)
                    .attr('height', height);
                const root = d3.hierarchy(data)
                    .sum(d => d.value || 0)
                    .sort((a, b) => b.value - a.value);
                d3.treemap()
                    .size([width, height])
                    .padding(1)(root);
                    const color = d3.scaleSequential()
                    .domain([0, d3.max(root.leaves(), d => d.value)])
                    .interpolator(d3.interpolate('#fff9c4', '#ff8a65'));
                svg.selectAll('rect')
                    .data(root.leaves())
                    .enter()
                    .append('rect')
                    .attr('x', d => d.x0)
                    .attr('y', d => d.y0)
                    .attr('width', d => d.x1 - d.x0)
                    .attr('height', d => d.y1 - d.y0)
                    .attr('fill', d => color(d.value))
                    .append('title')
                    .text(d => `${d.data.name}: ${d.value} changes`);
                svg.selectAll('text')
                    .data(root.leaves())
                    .enter()
                    .append('text')
                    .attr('x', d => d.x0 + 5)
                    .attr('y', d => d.y0 + 15)
                    .text(d => d.data.name)
                    .attr('font-size', '10px')
                    .attr('fill', '#090900');
            };
            onMounted(() => {
                fetchData();
            });

            return {
                heatmap,
            };
        },
    };
</script>

<style scoped>
    .container {
    display: flex;
    flex-direction: column;
    align-items: center;
    }

    i {
    font-style: italic;
    font-weight: 100;
    font-size: smaller;
    padding: 20px;
    }

    .codebase-heatmap {
        max-width: 800px;
        margin: 0 auto;
    }

    svg {
    width: 100%;
    }

        .rect-text {
    fill: #000;
    font-size: 10px;
    }

    .legend {
    display: flex;
    justify-content: center;
    margin-top: 10px;
    gap: 15px;
    }

    .color-box {
    display: inline-block;
    width: 15px;
    height: 15px;
    margin-right: 5px;
    vertical-align: middle;
    }

    .color-box.low {
    background-color: #fff9c4; /* Light Yellow */
    }

    .color-box.medium {
    background-color: #ffe0b2; /* Light Orange */
    }

    .color-box.high {
    background-color: #ff8a65; /* Orange */
    }
</style>
