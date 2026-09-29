package com.docscannerrn

import com.facebook.react.common.MapBuilder
import com.facebook.react.uimanager.SimpleViewManager
import com.facebook.react.uimanager.ThemedReactContext
import com.facebook.react.uimanager.annotations.ReactProp

/**
 * Exposes [DocumentScannerView] to React Native as the "DocumentScannerView"
 * component. Capture is triggered by incrementing the `captureTrigger` prop
 * (a nonce), which avoids the view-command plumbing. Results come back via the
 * `onDocumentCaptured` / `onScannerError` direct events.
 */
class DocumentScannerViewManager : SimpleViewManager<DocumentScannerView>() {

    override fun getName(): String = "DocumentScannerView"

    override fun createViewInstance(reactContext: ThemedReactContext): DocumentScannerView =
        DocumentScannerView(reactContext)

    @ReactProp(name = "captureTrigger", defaultInt = 0)
    fun setCaptureTrigger(view: DocumentScannerView, value: Int) {
        view.onCaptureTriggerChanged(value)
    }

    override fun getExportedCustomDirectEventTypeConstants(): MutableMap<String, Any> =
        MapBuilder.builder<String, Any>()
            .put("onDocumentCaptured", MapBuilder.of("registrationName", "onDocumentCaptured"))
            .put("onScannerError", MapBuilder.of("registrationName", "onScannerError"))
            .build()
}
