import { app } from "../../scripts/app.js";

/**
 * Cleanly show or hide a LiteGraph widget without destroying its reference or value.
 * Safely collapses the widget in LiteGraph without triggering ComfyUI's "convert to input slot" auto-behavior.
 */
function toggleWidget(widget, visible) {
    if (!widget) return;
    if (!widget.options) widget.options = {};

    if (visible) {
        widget.hidden = false;
        widget.options.hidden = false;

        if (widget.origType !== undefined) {
            widget.type = widget.origType;
            delete widget.origType;
        }
        if (widget.origComputeSize !== undefined) {
            widget.computeSize = widget.origComputeSize;
            delete widget.origComputeSize;
        } else {
            delete widget.computeSize;
        }
        if (widget._hiddenDrawHooked) {
            if (widget._origDraw !== undefined) {
                widget.draw = widget._origDraw;
            } else {
                delete widget.draw;
            }
            delete widget._hiddenDrawHooked;
        }
        if (widget.element) {
            widget.element.style.display = "";
        }
    } else {
        widget.hidden = true;
        widget.options.hidden = true;

        if (widget.origType === undefined) {
            widget.origType = widget.type;
        }
        if (widget.origComputeSize === undefined) {
            widget.origComputeSize = widget.computeSize;
        }
        if (!widget._hiddenDrawHooked) {
            widget._origDraw = widget.hasOwnProperty("draw") ? widget.draw : undefined;
            widget._hiddenDrawHooked = true;
        }
        widget.type = "hidden";
        widget.computeSize = () => [0, -4]; // -4 cancels out ComfyUI's hardcoded 4px widget padding
        widget.draw = () => {};
        if (widget.element) {
            widget.element.style.display = "none";
        }
    }
}

/**
 * Recomputes node bounding box to fit currently visible widgets snugly.
 */
function resizeNode(node) {
    if (!node) return;
    const sz = node.computeSize();
    node.setSize([Math.max(node.size[0] || 0, sz[0]), sz[1]]);
    node.setDirtyCanvas(true, true);
}

// --------------------------------------------------------------------------
// Helper to hook widget callbacks and node lifecycle events
// --------------------------------------------------------------------------
function attachDynamicWidgetHandler(node, targetClass, watchedWidgetNames, updateFn) {
    if (node.comfyClass !== targetClass) return;

    if (node.widgets && watchedWidgetNames.length > 0) {
        for (const name of watchedWidgetNames) {
            const w = node.widgets.find(widget => widget.name === name);
            if (w && !w._h3Watched) {
                w._h3Watched = true;
                const origCallback = w.callback;
                w.callback = function(value) {
                    if (origCallback) origCallback.apply(this, arguments);
                    updateFn(node);
                };
            }
        }
    }

    if (!node._h3LifecycleAttached) {
        node._h3LifecycleAttached = true;

        const origOnConnectionsChange = node.onConnectionsChange;
        node.onConnectionsChange = function(type, index, connected, link_info) {
            if (origOnConnectionsChange) origOnConnectionsChange.apply(this, arguments);
            updateFn(node);
        };

        const origOnConfigure = node.onConfigure;
        node.onConfigure = function() {
            if (origOnConfigure) origOnConfigure.apply(this, arguments);
            updateFn(node);
        };
    }

    updateFn(node);
    setTimeout(() => updateFn(node), 20);
    setTimeout(() => updateFn(node), 100);
}

// --------------------------------------------------------------------------
// 1. H3ChunkVideoDescriber UI Handler
// --------------------------------------------------------------------------
function updateChunkDescriberWidgets(node) {
    if (!node || !node.widgets) return;
    const saveWidget = node.widgets.find(w => w.name === "save_prompts_to_output");
    const pathWidget = node.widgets.find(w => w.name === "prompt_output_path");
    if (pathWidget && saveWidget) {
        toggleWidget(pathWidget, Boolean(saveWidget.value));
    }
    resizeNode(node);
}

// --------------------------------------------------------------------------
// 2. H3LatentTiledKSampler UI Handler
// --------------------------------------------------------------------------
function updateTiledKSamplerWidgets(node) {
    if (!node || !node.widgets) return;
    const debugWidget = node.widgets.find(w => w.name === "debug_decode_chunks");
    const dirWidget = node.widgets.find(w => w.name === "debug_chunk_output_dir");
    if (dirWidget && debugWidget) {
        toggleWidget(dirWidget, Boolean(debugWidget.value));
    }
    resizeNode(node);
}

// --------------------------------------------------------------------------
// 3. H3VideoSave UI Handler
// --------------------------------------------------------------------------
function updateVideoSaveWidgets(node) {
    if (!node || !node.widgets) return;
    const copyWidget = node.widgets.find(w => w.name === "save_latent_copy");
    const dirWidget = node.widgets.find(w => w.name === "latent_output_dir");
    const sepWidget = node.widgets.find(w => w.name === "save_latents_separately");

    const isCopyActive = Boolean(copyWidget?.value);
    if (dirWidget) toggleWidget(dirWidget, isCopyActive);
    if (sepWidget) toggleWidget(sepWidget, isCopyActive);
    resizeNode(node);
}

// --------------------------------------------------------------------------
// 4. H3LatentCanvasUpscale & H3TileCalculator Custom Aspect Ratio Handler
// --------------------------------------------------------------------------
function updateAspectRatioWidgets(node) {
    if (!node || !node.widgets) return;
    const aspectWidget = node.widgets.find(w => w.name === "output_aspect_ratio" || w.name === "aspect_ratio");
    const customWidget = node.widgets.find(w => w.name === "custom_aspect_ratio");
    if (customWidget && aspectWidget) {
        toggleWidget(customWidget, String(aspectWidget.value) === "Custom");
    }
    resizeNode(node);
}

// --------------------------------------------------------------------------
// 5. H3ReferenceAssetBundle Dynamic Input Slots Handler
// --------------------------------------------------------------------------
function updateReferenceBundleSlots(node) {
    if (!node || !node.inputs) return;
    if (node._updatingSlots) return;
    node._updatingSlots = true;

    try {
        const categories = [
            { prefix: "picture", max: 9, type: "IMAGE" },
            { prefix: "video", max: 3, type: "IMAGE,VIDEO" },
            { prefix: "audio", max: 3, type: "AUDIO" }
        ];

        let totalConnected = 0;
        for (const input of node.inputs) {
            if (input && input.link != null) {
                totalConnected++;
            }
        }

        for (const cat of categories) {
            let maxConnectedIdx = 0;
            for (const input of node.inputs) {
                if (input && input.name && input.name.startsWith(cat.prefix + "_")) {
                    const parts = input.name.split("_");
                    const num = parseInt(parts[1], 10);
                    if (!isNaN(num) && input.link != null) {
                        if (num > maxConnectedIdx) {
                            maxConnectedIdx = num;
                        }
                    }
                }
            }

            let desiredCount = Math.max(1, maxConnectedIdx + 1);
            if (desiredCount > cat.max) {
                desiredCount = cat.max;
            }
            while (desiredCount > maxConnectedIdx && (totalConnected + 1) > 12) {
                desiredCount--;
            }

            for (let i = node.inputs.length - 1; i >= 0; i--) {
                const inp = node.inputs[i];
                if (inp && inp.name && inp.name.startsWith(cat.prefix + "_")) {
                    const num = parseInt(inp.name.split("_")[1], 10);
                    if (!isNaN(num) && num > desiredCount && inp.link == null) {
                        node.removeInput(i);
                    }
                }
            }

            for (let s = 1; s <= desiredCount; s++) {
                const slotName = `${cat.prefix}_${s}`;
                const exists = node.inputs.some(inp => inp && inp.name === slotName);
                if (!exists) {
                    node.addInput(slotName, cat.type);
                }
            }
        }

        resizeNode(node);
    } finally {
        node._updatingSlots = false;
    }
}

function setupDynamicNode(node) {
    if (!node) return;
    switch (node.comfyClass) {
        case "H3ReferenceAssetBundle":
            attachDynamicWidgetHandler(node, "H3ReferenceAssetBundle", [], updateReferenceBundleSlots);
            break;

        case "H3ChunkVideoDescriber":
            attachDynamicWidgetHandler(node, "H3ChunkVideoDescriber", ["save_prompts_to_output"], updateChunkDescriberWidgets);
            break;

        case "H3LatentTiledKSampler":
            attachDynamicWidgetHandler(node, "H3LatentTiledKSampler", ["debug_decode_chunks"], updateTiledKSamplerWidgets);
            break;

        case "H3VideoSave":
            attachDynamicWidgetHandler(node, "H3VideoSave", ["save_latent_copy"], updateVideoSaveWidgets);
            break;

        case "H3LatentCanvasUpscale":
            attachDynamicWidgetHandler(node, "H3LatentCanvasUpscale", ["output_aspect_ratio"], updateAspectRatioWidgets);
            break;

        case "H3TileCalculator":
            attachDynamicWidgetHandler(node, "H3TileCalculator", ["aspect_ratio"], updateAspectRatioWidgets);
            break;
    }
}

// --------------------------------------------------------------------------
// Register Global ComfyUI Extension
// --------------------------------------------------------------------------
app.registerExtension({
    name: "ComfyUI-H3-TiledUpscale.DynamicUI",
    async nodeCreated(node) {
        setupDynamicNode(node);
    },
    async loadedGraphNode(node) {
        setupDynamicNode(node);
    },
    async afterConfigureGraph() {
        if (app.graph && app.graph._nodes) {
            for (const node of app.graph._nodes) {
                setupDynamicNode(node);
            }
        }
    }
});
