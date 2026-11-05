plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}
android {
    namespace = "com.peters.voicecontrol"
    compileSdk = 35
    defaultConfig {
        applicationId = "com.peters.voicecontrol"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1"
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    packaging { resources { excludes += "/META-INF/{AL2.0,LGPL2.1}" } }
    androidResources { noCompress += "tflite" }
}
dependencies {
    implementation("com.google.ai.edge.litert:litert:1.4.2")
    implementation("com.github.wendykierp:JTransforms:3.1")
}
