plugins {
    application
}

repositories { mavenCentral() }

dependencies {
    implementation(project(":lib"))   // <-- add this line
    // keep your other dependencies (JUnit, etc.)
}

application {
    // Set to your actual main class (fully qualified)
    mainClass = "com.example.app.Main"
}

java {
    toolchain { languageVersion.set(JavaLanguageVersion.of(21)) }
}

tasks.test { useJUnitPlatform() }
