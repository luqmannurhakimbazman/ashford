import AppKit
import Foundation

let args = CommandLine.arguments
if args.count != 5 {
    fputs("usage: render dashboard receipt terminal output-dir\n", stderr)
    exit(2)
}
let dashboardPath = args[1]
let receiptPath = args[2]
let terminalPath = args[3]
let outputDir = args[4]
let dashboard = try String(contentsOfFile: dashboardPath, encoding: .utf8)
let receipt = try String(contentsOfFile: receiptPath, encoding: .utf8)
let terminal = try String(contentsOfFile: terminalPath, encoding: .utf8)
try FileManager.default.createDirectory(atPath: outputDir, withIntermediateDirectories: true)

let width: CGFloat = 1600
let height: CGFloat = 1000
let sidebarWidth: CGFloat = 330
let bg = NSColor(calibratedRed: 0.105, green: 0.105, blue: 0.125, alpha: 1)
let panel = NSColor(calibratedRed: 0.145, green: 0.145, blue: 0.170, alpha: 1)
let editor = NSColor(calibratedRed: 0.125, green: 0.125, blue: 0.150, alpha: 1)
let text = NSColor(calibratedRed: 0.86, green: 0.86, blue: 0.90, alpha: 1)
let muted = NSColor(calibratedRed: 0.60, green: 0.60, blue: 0.67, alpha: 1)
let accent = NSColor(calibratedRed: 0.66, green: 0.52, blue: 0.90, alpha: 1)
let green = NSColor(calibratedRed: 0.43, green: 0.82, blue: 0.59, alpha: 1)
let amber = NSColor(calibratedRed: 0.95, green: 0.72, blue: 0.38, alpha: 1)

func fill(_ color: NSColor, _ rect: NSRect) {
    color.setFill(); NSBezierPath(rect: rect).fill()
}

func draw(_ value: String, rect: NSRect, font: NSFont, color: NSColor = text, lineSpacing: CGFloat = 2, alignment: NSTextAlignment = .left) {
    let style = NSMutableParagraphStyle()
    style.lineBreakMode = .byWordWrapping
    style.lineSpacing = lineSpacing
    style.alignment = alignment
    (value as NSString).draw(in: rect, withAttributes: [.font: font, .foregroundColor: color, .paragraphStyle: style])
}

func clean(_ value: String) -> String {
    var s = value
    s = s.replacingOccurrences(of: "**", with: "")
    s = s.replacingOccurrences(of: "`", with: "")
    s = s.replacingOccurrences(of: "[[sessions/session-demo-001.md|session-demo-001]]", with: "session-demo-001 ↗")
    return s
}

func sidebar(selected: String) {
    fill(panel, NSRect(x: 0, y: 0, width: sidebarWidth, height: height))
    draw("DUNK VAULT", rect: NSRect(x: 28, y: 925, width: 270, height: 28), font: .systemFont(ofSize: 13, weight: .semibold), color: muted)
    draw("⌄  domains", rect: NSRect(x: 28, y: 880, width: 270, height: 26), font: .systemFont(ofSize: 16, weight: .medium))
    draw("⌄  bayesian-forecasting-0b6072a9", rect: NSRect(x: 48, y: 842, width: 270, height: 26), font: .systemFont(ofSize: 14, weight: .medium), color: text)
    let files = ["profile.yaml", "events.jsonl", "state.json", "dashboard.md", "⌄  sessions", "   session-demo-001.md"]
    var y: CGFloat = 804
    for file in files {
        if file.trimmingCharacters(in: .whitespaces) == selected {
            let selection = NSBezierPath(roundedRect: NSRect(x: 38, y: y - 5, width: 280, height: 31), xRadius: 5, yRadius: 5)
            NSColor(calibratedRed: 0.28, green: 0.23, blue: 0.39, alpha: 1).setFill(); selection.fill()
        }
        let icon = file.contains("sessions") ? "▾" : (file.hasSuffix(".md") ? "◇" : "≡")
        draw("\(icon)  \(file)", rect: NSRect(x: 58, y: y, width: 245, height: 24), font: .systemFont(ofSize: 14), color: file.trimmingCharacters(in: .whitespaces) == selected ? NSColor.white : text)
        y -= 36
    }
    draw("Filesystem is canonical\nObsidian plugin: not required", rect: NSRect(x: 28, y: 40, width: 270, height: 52), font: .systemFont(ofSize: 12), color: muted, lineSpacing: 5)
}

func drawMarkdown(_ markdown: String, selected: String, output: String) throws {
    let image = NSImage(size: NSSize(width: width, height: height))
    image.lockFocus()
    fill(bg, NSRect(x: 0, y: 0, width: width, height: height))
    sidebar(selected: selected)
    fill(editor, NSRect(x: sidebarWidth, y: 0, width: width - sidebarWidth, height: height))
    fill(panel, NSRect(x: sidebarWidth, y: 940, width: width - sidebarWidth, height: 60))
    draw("\(selected)    ×", rect: NSRect(x: sidebarWidth + 30, y: 958, width: 500, height: 24), font: .systemFont(ofSize: 14, weight: .medium))
    draw("Reading view", rect: NSRect(x: width - 155, y: 958, width: 120, height: 24), font: .systemFont(ofSize: 12), color: muted, alignment: .right)

    var y: CGFloat = 902
    let x = sidebarWidth + 60
    let contentWidth = width - sidebarWidth - 120
    var inTable = false
    for raw in markdown.split(separator: "\n", omittingEmptySubsequences: false).map(String.init) {
        let line = clean(raw)
        if line.isEmpty { y -= 9; continue }
        if line.hasPrefix("# ") {
            draw(String(line.dropFirst(2)), rect: NSRect(x: x, y: y - 9, width: contentWidth, height: 42), font: .systemFont(ofSize: 30, weight: .bold), color: NSColor.white)
            y -= 48; inTable = false
        } else if line.hasPrefix("## ") {
            draw(String(line.dropFirst(3)), rect: NSRect(x: x, y: y, width: contentWidth, height: 30), font: .systemFont(ofSize: 20, weight: .semibold), color: accent)
            y -= 34; inTable = false
        } else if line.hasPrefix("> [!warning]") {
            let message = line.replacingOccurrences(of: "> [!warning] ", with: "⚠  ")
            let box = NSBezierPath(roundedRect: NSRect(x: x, y: y - 4, width: contentWidth, height: 34), xRadius: 5, yRadius: 5)
            NSColor(calibratedRed: 0.27, green: 0.22, blue: 0.13, alpha: 1).setFill(); box.fill()
            draw(message, rect: NSRect(x: x + 12, y: y + 4, width: contentWidth - 24, height: 22), font: .systemFont(ofSize: 12, weight: .medium), color: amber)
            y -= 42; inTable = false
        } else if line.hasPrefix("|---") {
            continue
        } else if line.hasPrefix("|") {
            let cells = line.split(separator: "|", omittingEmptySubsequences: true).map { $0.trimmingCharacters(in: .whitespaces) }
            let rendered = cells.joined(separator: "   │   ")
            draw(rendered, rect: NSRect(x: x + 8, y: y, width: contentWidth - 16, height: 24), font: .monospacedSystemFont(ofSize: 11.5, weight: inTable ? .regular : .semibold), color: inTable ? text : muted)
            y -= 24; inTable = true
        } else {
            let color = line.contains("not measured") || line.contains("Next review") ? amber : text
            draw(line, rect: NSRect(x: x, y: y, width: contentWidth, height: 25), font: .systemFont(ofSize: 13.5), color: color)
            y -= 24; inTable = false
        }
        if y < 42 { break }
    }
    draw("Actual dln-store fixture output • rendered 2026-08-18", rect: NSRect(x: x, y: 14, width: contentWidth, height: 20), font: .systemFont(ofSize: 11), color: muted, alignment: .right)
    image.unlockFocus()
    let rep = NSBitmapImageRep(data: image.tiffRepresentation!)!
    try rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: output))
}

func drawTerminal(_ transcript: String, output: String) throws {
    let image = NSImage(size: NSSize(width: width, height: height))
    image.lockFocus()
    fill(NSColor(calibratedRed: 0.055, green: 0.065, blue: 0.080, alpha: 1), NSRect(x: 0, y: 0, width: width, height: height))
    fill(NSColor(calibratedRed: 0.12, green: 0.13, blue: 0.15, alpha: 1), NSRect(x: 0, y: 944, width: width, height: 56))
    let colors = [NSColor.systemRed, NSColor.systemYellow, NSColor.systemGreen]
    for (i, color) in colors.enumerated() {
        color.setFill(); NSBezierPath(ovalIn: NSRect(x: 24 + CGFloat(i) * 25, y: 964, width: 13, height: 13)).fill()
    }
    draw("dunk issue #1 — local validation", rect: NSRect(x: 570, y: 961, width: 460, height: 22), font: .monospacedSystemFont(ofSize: 13, weight: .medium), color: muted, alignment: .center)
    var y: CGFloat = 915
    for raw in transcript.split(separator: "\n", omittingEmptySubsequences: false).map(String.init) {
        let isCommand = raw.hasPrefix("$")
        let isSuccess = raw.contains("passed") || raw.contains("Validation passed") || raw.contains("hashes_unchanged=true") || raw.contains("\"status\":\"noop\"") || raw.contains("revision_before=1 revision_after=1")
        let color = isCommand ? accent : (isSuccess ? green : text)
        draw(raw, rect: NSRect(x: 42, y: y, width: width - 84, height: 20), font: .monospacedSystemFont(ofSize: 12.2, weight: isCommand ? .semibold : .regular), color: color, lineSpacing: 0)
        y -= 22
        if y < 24 { break }
    }
    image.unlockFocus()
    let rep = NSBitmapImageRep(data: image.tiffRepresentation!)!
    try rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: output))
}

try drawMarkdown(dashboard, selected: "dashboard.md", output: outputDir + "/obsidian-dashboard.png")
try drawMarkdown(receipt, selected: "session-demo-001.md", output: outputDir + "/session-receipt.png")
try drawTerminal(terminal, output: outputDir + "/terminal-validation.png")
