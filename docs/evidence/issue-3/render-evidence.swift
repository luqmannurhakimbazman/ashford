import AppKit
import Foundation

let args = CommandLine.arguments
if args.count != 8 {
    fputs("usage: render intake dashboard session terminal output-dir render-date domain-id\n", stderr)
    exit(2)
}
let intake = try String(contentsOfFile: args[1], encoding: .utf8)
let dashboard = try String(contentsOfFile: args[2], encoding: .utf8)
let session = try String(contentsOfFile: args[3], encoding: .utf8)
let terminal = try String(contentsOfFile: args[4], encoding: .utf8)
let outputDir = args[5]
let renderDate = args[6]
let domainID = args[7]
try FileManager.default.createDirectory(atPath: outputDir, withIntermediateDirectories: true)

let width: CGFloat = 1600
let height: CGFloat = 1000
let sidebarWidth: CGFloat = 340
let background = NSColor(calibratedRed: 0.105, green: 0.105, blue: 0.125, alpha: 1)
let panel = NSColor(calibratedRed: 0.145, green: 0.145, blue: 0.170, alpha: 1)
let editor = NSColor(calibratedRed: 0.125, green: 0.125, blue: 0.150, alpha: 1)
let text = NSColor(calibratedRed: 0.86, green: 0.86, blue: 0.90, alpha: 1)
let muted = NSColor(calibratedRed: 0.60, green: 0.60, blue: 0.67, alpha: 1)
let accent = NSColor(calibratedRed: 0.66, green: 0.52, blue: 0.90, alpha: 1)
let green = NSColor(calibratedRed: 0.43, green: 0.82, blue: 0.59, alpha: 1)
let amber = NSColor(calibratedRed: 0.95, green: 0.72, blue: 0.38, alpha: 1)

func canvas() -> NSBitmapImageRep {
    let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: Int(width), pixelsHigh: Int(height), bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .calibratedRGB, bytesPerRow: 0, bitsPerPixel: 0)!
    rep.size = NSSize(width: width, height: height)
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)!
    return rep
}

func finish(_ rep: NSBitmapImageRep, _ path: String) throws {
    NSGraphicsContext.current?.flushGraphics()
    NSGraphicsContext.restoreGraphicsState()
    try rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: path))
}

func fill(_ color: NSColor, _ rect: NSRect) {
    color.setFill()
    NSBezierPath(rect: rect).fill()
}

func draw(_ value: String, _ rect: NSRect, _ font: NSFont, color: NSColor = text, alignment: NSTextAlignment = .left) {
    let style = NSMutableParagraphStyle()
    style.lineBreakMode = .byTruncatingTail
    style.alignment = alignment
    (value as NSString).draw(in: rect, withAttributes: [.font: font, .foregroundColor: color, .paragraphStyle: style])
}

func clean(_ value: String) -> String {
    value.replacingOccurrences(of: "**", with: "").replacingOccurrences(of: "`", with: "").replacingOccurrences(of: "&amp;", with: "&")
}

func sidebar(_ selected: String) {
    fill(panel, NSRect(x: 0, y: 0, width: sidebarWidth, height: height))
    draw("DUNK ISSUE #3 FIXTURE", NSRect(x: 26, y: 934, width: 285, height: 24), .systemFont(ofSize: 13, weight: .semibold), color: muted)
    draw("⌄  domains", NSRect(x: 26, y: 890, width: 285, height: 25), .systemFont(ofSize: 16, weight: .medium))
    draw("⌄  \(domainID)", NSRect(x: 44, y: 852, width: 285, height: 25), .systemFont(ofSize: 13, weight: .medium))
    let files = ["profile.yaml", "events.jsonl", "state.json", "dashboard.md", "⌄  syllabus", "   intake receipt.md", "⌄  sessions", "   grounded session.md"]
    var y: CGFloat = 812
    for file in files {
        let label = file.trimmingCharacters(in: .whitespaces)
        if label == selected {
            NSColor(calibratedRed: 0.28, green: 0.23, blue: 0.39, alpha: 1).setFill()
            NSBezierPath(roundedRect: NSRect(x: 34, y: y - 5, width: 294, height: 31), xRadius: 5, yRadius: 5).fill()
        }
        draw("◇  \(file)", NSRect(x: 54, y: y, width: 265, height: 23), .systemFont(ofSize: 13), color: label == selected ? .white : text)
        y -= 35
    }
    draw("Rendered real dln-store artifacts", NSRect(x: 26, y: 34, width: 285, height: 22), .systemFont(ofSize: 12), color: muted)
}

func markdownLines(_ markdown: String, kind: String) -> [String] {
    let all = markdown.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
    if kind != "intake" { return all }
    return Array(all.prefix(31))
}

func markdown(_ source: String, selected: String, kind: String, output: String) throws {
    let rep = canvas()
    fill(background, NSRect(x: 0, y: 0, width: width, height: height))
    sidebar(selected)
    fill(editor, NSRect(x: sidebarWidth, y: 0, width: width - sidebarWidth, height: height))
    fill(panel, NSRect(x: sidebarWidth, y: 944, width: width - sidebarWidth, height: 56))
    draw("\(selected)    ×", NSRect(x: sidebarWidth + 28, y: 962, width: 600, height: 22), .systemFont(ofSize: 14, weight: .medium))
    draw("Rendered fixture output", NSRect(x: width - 260, y: 962, width: 220, height: 22), .systemFont(ofSize: 12), color: muted, alignment: .right)

    var y: CGFloat = 906
    let x = sidebarWidth + 48
    let contentWidth = width - sidebarWidth - 96
    for raw in markdownLines(source, kind: kind) {
        let line = clean(raw)
        if line.isEmpty { y -= 10; continue }
        if line.hasPrefix("# ") {
            draw(String(line.dropFirst(2)), NSRect(x: x, y: y - 9, width: contentWidth, height: 42), .systemFont(ofSize: 28, weight: .bold), color: .white)
            y -= 48
        } else if line.hasPrefix("### ") {
            draw(String(line.dropFirst(4)), NSRect(x: x, y: y, width: contentWidth, height: 26), .systemFont(ofSize: 16, weight: .semibold), color: amber)
            y -= 29
        } else if line.hasPrefix("## ") {
            draw(String(line.dropFirst(3)), NSRect(x: x, y: y, width: contentWidth, height: 30), .systemFont(ofSize: 20, weight: .semibold), color: accent)
            y -= 34
        } else if line.hasPrefix("> [!warning]") {
            NSColor(calibratedRed: 0.27, green: 0.22, blue: 0.13, alpha: 1).setFill()
            NSBezierPath(roundedRect: NSRect(x: x, y: y - 5, width: contentWidth, height: 31), xRadius: 5, yRadius: 5).fill()
            draw("⚠  " + line.replacingOccurrences(of: "> [!warning] ", with: ""), NSRect(x: x + 10, y: y + 2, width: contentWidth - 20, height: 21), .systemFont(ofSize: 12, weight: .medium), color: amber)
            y -= 39
        } else if line.hasPrefix("|---") {
            continue
        } else {
            let highlight = line.contains("approved") || line.contains("Unresolved") || line.contains("SHA-256") || line.contains("Approval")
            draw(line, NSRect(x: x, y: y, width: contentWidth, height: 23), line.hasPrefix("|") ? .monospacedSystemFont(ofSize: 11.3, weight: .regular) : .systemFont(ofSize: 13.2), color: highlight ? amber : text)
            y -= 23
        }
        if y < 48 { break }
    }
    draw("Actual dln-store fixture output • rendered \(renderDate)", NSRect(x: x, y: 15, width: contentWidth, height: 20), .systemFont(ofSize: 11), color: muted, alignment: .right)
    try finish(rep, output)
}

func terminalImage(_ source: String, output: String) throws {
    let rep = canvas()
    fill(NSColor(calibratedRed: 0.055, green: 0.065, blue: 0.080, alpha: 1), NSRect(x: 0, y: 0, width: width, height: height))
    fill(panel, NSRect(x: 0, y: 944, width: width, height: 56))
    draw("dunk issue #3 — grounding validation", NSRect(x: 520, y: 962, width: 560, height: 22), .monospacedSystemFont(ofSize: 13, weight: .medium), color: muted, alignment: .center)
    var y: CGFloat = 910
    for raw in source.split(separator: "\n", omittingEmptySubsequences: false).map(String.init) {
        let command = raw.hasPrefix("$")
        let success = raw.contains("passed") || raw.contains("Validation passed") || raw.contains("hashes_unchanged=true") || raw.contains("digest_mismatch_at_expected_size=true") || raw.contains("degradation_preserved=true")
        draw(raw, NSRect(x: 38, y: y, width: width - 76, height: 20), .monospacedSystemFont(ofSize: 12.0, weight: command ? .semibold : .regular), color: command ? accent : (success ? green : text))
        y -= 22
        if y < 24 { break }
    }
    try finish(rep, output)
}

try markdown(intake, selected: "intake receipt.md", kind: "intake", output: outputDir + "/st5201x-intake-preapproval.png")
try markdown(dashboard, selected: "dashboard.md", kind: "dashboard", output: outputDir + "/st5201x-approved-dashboard.png")
try markdown(session, selected: "grounded session.md", kind: "session", output: outputDir + "/st5201x-grounded-session.png")
try terminalImage(terminal, output: outputDir + "/terminal-validation.png")
