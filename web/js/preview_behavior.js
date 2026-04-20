import { app } from "../../../scripts/app.js";

const LOAD_RETINAFACE_NODE = "Flux FaceIR Load RetinaFace";
const RESTORE_FACE_NODE = "Flux FaceIR Restore Face";

function clearNodePreview(node) {
	if (!node) {
		return;
	}
	node.imgs = null;
	node.imageIndex = null;
	node.overIndex = null;
	node.setDirtyCanvas?.(true, true);
	app.graph?.setDirtyCanvas?.(true, true);
}

app.registerExtension({
	name: "FluxFaceIR.PreviewBehavior",

	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData.name === LOAD_RETINAFACE_NODE && !nodeType.prototype.__fluxFaceIRLoadPreviewPatched) {
			nodeType.prototype.__fluxFaceIRLoadPreviewPatched = true;

			const onNodeCreated = nodeType.prototype.onNodeCreated;
			const onConfigure = nodeType.prototype.onConfigure;
			const onExecuted = nodeType.prototype.onExecuted;

			nodeType.prototype.onNodeCreated = function () {
				onNodeCreated?.apply(this, arguments);
				clearNodePreview(this);
			};

			nodeType.prototype.onConfigure = function () {
				onConfigure?.apply(this, arguments);
				clearNodePreview(this);
			};

			nodeType.prototype.onExecuted = function () {
				onExecuted?.apply(this, arguments);
				clearNodePreview(this);
			};
		}

		if (nodeData.name === RESTORE_FACE_NODE && !nodeType.prototype.__fluxFaceIRRestorePreviewPatched) {
			nodeType.prototype.__fluxFaceIRRestorePreviewPatched = true;

			const onExecuted = nodeType.prototype.onExecuted;

			nodeType.prototype.onExecuted = function () {
				onExecuted?.apply(this, arguments);
				this.imageIndex = 0;
				this.overIndex = null;
				this.setDirtyCanvas?.(true, true);
				app.graph?.setDirtyCanvas?.(true, true);
			};
		}
	},
});
