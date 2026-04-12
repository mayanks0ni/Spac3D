module.exports = [
"[project]/src/app/page.tsx [app-ssr] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "default",
    ()=>Home
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react-jsx-dev-runtime.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/dist/server/route-modules/app-page/vendored/ssr/react.js [app-ssr] (ecmascript)");
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$navigation$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/navigation.js [app-ssr] (ecmascript)");
"use client";
;
;
;
function Home() {
    const [file, setFile] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useState"])(null);
    const [preview, setPreview] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useState"])(null);
    const [status, setStatus] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useState"])("idle");
    const [errorMsg, setErrorMsg] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useState"])(null);
    const [isDragOver, setIsDragOver] = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useState"])(false);
    const router = (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$navigation$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useRouter"])();
    (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["useEffect"])(()=>{
        if (file) {
            const url = URL.createObjectURL(file);
            setPreview(url);
            return ()=>URL.revokeObjectURL(url);
        }
    }, [
        file
    ]);
    const handleFile = (selectedFile)=>{
        if (!selectedFile) return;
        if (!selectedFile.type.startsWith("image/")) {
            setErrorMsg("Please upload an image file.");
            return;
        }
        setFile(selectedFile);
        setErrorMsg(null); // Reset error state on new file
    };
    const handleDrop = (e)=>{
        e.preventDefault();
        setIsDragOver(false);
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    };
    const processImage = async ()=>{
        if (!file) return;
        setStatus("loading");
        setErrorMsg(null);
        try {
            const formData = new FormData();
            formData.append("image", file);
            const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
            let res;
            try {
                res = await fetch(`${apiUrl}/process`, {
                    method: "POST",
                    body: formData
                });
            } catch (err) {
                setStatus("idle");
                console.error(err);
                setErrorMsg(`Network Error: ${err instanceof Error ? err.message : String(err)} | URL: ${apiUrl}/process`);
                return;
            }
            if (!res.ok) {
                const errText = await res.text();
                setStatus("idle");
                setErrorMsg(`API Error (${res.status}): ${errText.substring(0, 150)}`);
                return;
            }
            const data = await res.json();
            if (data.status === "success") {
                sessionStorage.setItem("ar_bg", data.inpainted_image_url);
                sessionStorage.setItem("ar_objects", JSON.stringify(data.detected_objects));
                router.push("/scene?mode=ar");
            } else {
                setStatus("idle");
                setErrorMsg(`Python Processing Error: ${data.message || "Unknown issue"}`);
            }
        } catch (err) {
            console.error(err);
            setStatus("idle");
            setErrorMsg("Something went wrong — try a clearer room photo");
        }
    };
    // Demo Dataset Handler
    const insertDemo = async (url)=>{
        try {
            const res = await fetch(url);
            const blob = await res.blob();
            const demoFile = new File([
                blob
            ], "demo-room.jpg", {
                type: "image/jpeg"
            });
            handleFile(demoFile);
        } catch (e) {
            setErrorMsg("Failed to load demo image from Unsplash.");
        }
    };
    return /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
        className: "container",
        children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
            className: "card",
            style: {
                maxWidth: "800px"
            },
            children: [
                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("h1", {
                    className: "title",
                    children: "Upload Room Image"
                }, void 0, false, {
                    fileName: "[project]/src/app/page.tsx",
                    lineNumber: 103,
                    columnNumber: 9
                }, this),
                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("p", {
                    className: "subtitle",
                    children: "Drag and drop your space photo to start analyzing."
                }, void 0, false, {
                    fileName: "[project]/src/app/page.tsx",
                    lineNumber: 104,
                    columnNumber: 9
                }, this),
                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                    className: `upload-box ${isDragOver ? "drag-over" : ""}`,
                    onDragOver: (e)=>{
                        e.preventDefault();
                        setIsDragOver(true);
                    },
                    onDragLeave: (e)=>{
                        e.preventDefault();
                        setIsDragOver(false);
                    },
                    onDrop: handleDrop,
                    onClick: ()=>document.getElementById("file-upload")?.click(),
                    style: {
                        cursor: "pointer",
                        padding: "2rem",
                        border: "2px dashed",
                        borderColor: isDragOver ? "var(--primary)" : "var(--border)",
                        borderRadius: "16px",
                        backgroundColor: isDragOver ? "rgba(99, 102, 241, 0.1)" : "var(--glass)",
                        transition: "all 0.3s ease",
                        marginBottom: "1rem"
                    },
                    children: [
                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("input", {
                            type: "file",
                            onChange: (e)=>e.target.files && handleFile(e.target.files[0]),
                            accept: "image/*",
                            className: "file-input",
                            id: "file-upload"
                        }, void 0, false, {
                            fileName: "[project]/src/app/page.tsx",
                            lineNumber: 123,
                            columnNumber: 11
                        }, this),
                        !preview ? /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                            style: {
                                color: "var(--text-muted)"
                            },
                            children: [
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                    style: {
                                        fontSize: "2.5rem",
                                        marginBottom: "1rem"
                                    },
                                    children: "🖼️"
                                }, void 0, false, {
                                    fileName: "[project]/src/app/page.tsx",
                                    lineNumber: 133,
                                    columnNumber: 16
                                }, this),
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                                    children: "Drag & Drop or Click to Select Image"
                                }, void 0, false, {
                                    fileName: "[project]/src/app/page.tsx",
                                    lineNumber: 134,
                                    columnNumber: 16
                                }, this)
                            ]
                        }, void 0, true, {
                            fileName: "[project]/src/app/page.tsx",
                            lineNumber: 132,
                            columnNumber: 13
                        }, this) : /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                            style: {
                                position: "relative",
                                width: "100%",
                                height: "240px",
                                borderRadius: "12px",
                                overflow: "hidden"
                            },
                            children: /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("img", {
                                src: preview,
                                alt: "Preview",
                                style: {
                                    width: "100%",
                                    height: "100%",
                                    objectFit: "cover"
                                }
                            }, void 0, false, {
                                fileName: "[project]/src/app/page.tsx",
                                lineNumber: 139,
                                columnNumber: 17
                            }, this)
                        }, void 0, false, {
                            fileName: "[project]/src/app/page.tsx",
                            lineNumber: 137,
                            columnNumber: 14
                        }, this)
                    ]
                }, void 0, true, {
                    fileName: "[project]/src/app/page.tsx",
                    lineNumber: 106,
                    columnNumber: 9
                }, this),
                errorMsg && /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                    style: {
                        padding: "16px",
                        background: "#fee2e2",
                        color: "#b91c1c",
                        borderRadius: "8px",
                        fontSize: "0.95rem",
                        marginBottom: "1.5rem",
                        fontWeight: 600
                    },
                    children: errorMsg
                }, void 0, false, {
                    fileName: "[project]/src/app/page.tsx",
                    lineNumber: 145,
                    columnNumber: 11
                }, this),
                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                    onClick: (e)=>{
                        e.stopPropagation();
                        processImage();
                    },
                    disabled: !file || status === "loading",
                    className: "upload-button",
                    children: status === "loading" ? "Analyzing room..." : "Analyze Room"
                }, void 0, false, {
                    fileName: "[project]/src/app/page.tsx",
                    lineNumber: 150,
                    columnNumber: 9
                }, this),
                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                    style: {
                        marginTop: "2rem",
                        borderTop: "1px solid var(--border)",
                        paddingTop: "1.5rem"
                    },
                    children: [
                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("h3", {
                            style: {
                                fontSize: "1rem",
                                color: "var(--text-muted)",
                                marginBottom: "1rem"
                            },
                            children: "Or try a clear room from our Demo Dataset:"
                        }, void 0, false, {
                            fileName: "[project]/src/app/page.tsx",
                            lineNumber: 160,
                            columnNumber: 11
                        }, this),
                        /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("div", {
                            style: {
                                display: "flex",
                                gap: "1rem",
                                justifyContent: "center",
                                flexWrap: "wrap"
                            },
                            children: [
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                    onClick: (e)=>{
                                        e.stopPropagation();
                                        insertDemo("https://images.unsplash.com/photo-1555041469-a586c61ea9bc?w=800&q=80");
                                    },
                                    style: {
                                        background: "transparent",
                                        border: "1px solid var(--primary)",
                                        color: "var(--primary)",
                                        padding: "0.5rem 1rem",
                                        borderRadius: "8px",
                                        cursor: "pointer",
                                        fontWeight: 600
                                    },
                                    children: "🛋️ Minimalist Living Room"
                                }, void 0, false, {
                                    fileName: "[project]/src/app/page.tsx",
                                    lineNumber: 162,
                                    columnNumber: 13
                                }, this),
                                /*#__PURE__*/ (0, __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$dist$2f$server$2f$route$2d$modules$2f$app$2d$page$2f$vendored$2f$ssr$2f$react$2d$jsx$2d$dev$2d$runtime$2e$js__$5b$app$2d$ssr$5d$__$28$ecmascript$29$__["jsxDEV"])("button", {
                                    onClick: (e)=>{
                                        e.stopPropagation();
                                        insertDemo("https://images.unsplash.com/photo-1505691938895-1758d7feb511?w=800&q=80");
                                    },
                                    style: {
                                        background: "transparent",
                                        border: "1px solid var(--primary)",
                                        color: "var(--primary)",
                                        padding: "0.5rem 1rem",
                                        borderRadius: "8px",
                                        cursor: "pointer",
                                        fontWeight: 600
                                    },
                                    children: "🛏️ Modern Bedroom"
                                }, void 0, false, {
                                    fileName: "[project]/src/app/page.tsx",
                                    lineNumber: 168,
                                    columnNumber: 13
                                }, this)
                            ]
                        }, void 0, true, {
                            fileName: "[project]/src/app/page.tsx",
                            lineNumber: 161,
                            columnNumber: 11
                        }, this)
                    ]
                }, void 0, true, {
                    fileName: "[project]/src/app/page.tsx",
                    lineNumber: 159,
                    columnNumber: 9
                }, this)
            ]
        }, void 0, true, {
            fileName: "[project]/src/app/page.tsx",
            lineNumber: 102,
            columnNumber: 7
        }, this)
    }, void 0, false, {
        fileName: "[project]/src/app/page.tsx",
        lineNumber: 101,
        columnNumber: 5
    }, this);
}
}),
];

//# sourceMappingURL=src_app_page_tsx_0ss2.w7._.js.map