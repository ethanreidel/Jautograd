plugins { `java-library` }

repositories { mavenCentral() }

dependencies {
    // ND4J backend with natives for your OS/arch (CPU). Good default.
    implementation("org.nd4j:nd4j-native-platform:1.0.0-M2.1")
    // If INDArray is in your public API, use api(...) instead of implementation(...)

    // JUnit 5 for tests (keeps Gradle 9 happy)
    testImplementation(platform("org.junit:junit-bom:5.11.3"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
    testImplementation("org.mockito:mockito-junit-jupiter:5.+")
}

java {
    toolchain { languageVersion.set(JavaLanguageVersion.of(21)) }
}

tasks.test { useJUnitPlatform() }
