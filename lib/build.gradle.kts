plugins {
    `java-library`
    // Add this if you want to run a JavaFX app from Gradle:
    application
    // JavaFX helper plugin
    id("org.openjfx.javafxplugin") version "0.1.0"
}

repositories { mavenCentral() }

dependencies {
    implementation("org.nd4j:nd4j-native-platform:1.0.0-M2.1")
    implementation("org.datavec:datavec-api:1.0.0-M2")
    implementation("org.deeplearning4j:deeplearning4j-core:1.0.0-beta6")

    // SmartGraph
    implementation("com.brunomnsilva:smartgraph:2.3.0")

    // JavaFX (match JavaFX to your JDK; JavaFX 21 ↔ JDK 21)
    implementation("org.openjfx:javafx-controls:21.0.1")
    implementation("com.opencsv:opencsv:5.9")
    // If you use other JavaFX bits, add them too:
    // implementation("org.openjfx:javafx-graphics:21.0.1")
    // implementation("org.openjfx:javafx-fxml:21.0.1")

    testImplementation(platform("org.junit:junit-bom:5.11.3"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
    testImplementation("org.mockito:mockito-junit-jupiter:5.+")
}

java {
    toolchain { languageVersion.set(JavaLanguageVersion.of(21)) }
}

val onlyTest: String? = findProperty("onlyTest") as String?
tasks.named<JavaCompile>("compileTestJava") {
    if (onlyTest != null) include("**/${onlyTest.replace('.', '/')}.java")
}



tasks.test { useJUnitPlatform() }

// JavaFX plugin configuration
javafx {
    // Use the JavaFX 21 LTS line because you’re on JDK 21
    version = "21.0.1"
    modules = listOf("javafx.controls")
}

