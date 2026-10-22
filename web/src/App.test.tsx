import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";

const imageFile = () => new File(["fixture"], "portrait.png", { type: "image/png" });

describe("LeicaShot editor", () => {
  beforeEach(() => {
    let urlCounter = 0;
    vi.stubGlobal("URL", {
      ...URL,
      createObjectURL: vi.fn(() => `blob:laika-${++urlCounter}`),
      revokeObjectURL: vi.fn(),
    });
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("uploads a photo to GAIC and shows selectable crop suggestions", async () => {
    const fetchMock = vi.fn(async () =>
      Response.json({
        width: 1200,
        height: 1800,
        candidate_count: 556,
        suggestions: [
          {
            crop: { left: 0.1, top: 0.05, right: 0.9, bottom: 0.95 },
            score: 0.82,
            improvement_over_original: 0.13,
          },
        ],
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);

    fireEvent.change(screen.getByLabelText("Choose a photograph"), {
      target: { files: [imageFile()] },
    });
    fireEvent.click(screen.getByRole("button", { name: /suggest a crop/i }));

    expect(await screen.findByText("Crop 1")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /crop 1/i }));
    expect(screen.getByRole("button", { name: /clear selected crop/i })).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/v1/compositions",
      expect.objectContaining({ method: "POST", body: expect.any(FormData) }),
    );
  });

  it("submits the selected edit, follows progress, and exposes the full download", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/api/v1/tasks") && init?.method === "POST") {
        return Response.json(
          {
            id: "task-1",
            status: "queued",
            stage: "queued",
            progress: 0,
            error: null,
            access_token: "secret-task-token",
            status_url: "/api/v1/tasks/task-1",
            cancel_url: "/api/v1/tasks/task-1/cancel",
          },
          { status: 202 },
        );
      }
      if (url.endsWith("/api/v1/tasks/task-1")) {
        return Response.json({
          id: "task-1",
          status: "completed",
          stage: "complete",
          progress: 1,
          error: null,
          width: 900,
          height: 1200,
          preview_url: "/api/v1/tasks/task-1/preview",
          download_url: "/api/v1/tasks/task-1/download",
        });
      }
      if (url.endsWith("/preview")) {
        return new Response(new Blob(["preview"], { type: "image/png" }));
      }
      return new Response(new Blob(["full image"], { type: "image/png" }));
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);

    fireEvent.change(screen.getByLabelText("Choose a photograph"), {
      target: { files: [imageFile()] },
    });
    fireEvent.click(screen.getByRole("button", { name: /process photo/i }));

    const downloadButton = await screen.findByRole(
      "button",
      { name: /download full png/i },
      { timeout: 3000 },
    );
    expect(downloadButton).toBeInTheDocument();
    await waitFor(() => {
      const statusCall = fetchMock.mock.calls.find(([input]) =>
        String(input).endsWith("/api/v1/tasks/task-1"),
      );
      expect(statusCall?.[1]?.headers).toEqual({
        "X-Laika-Task-Token": "secret-task-token",
      });
    });
  });

  it("sends multiple subject prompts and the selected matte tile size", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/api/v1/tasks") && init?.method === "POST") {
        return Response.json(
          {
            id: "subject-task",
            status: "queued",
            stage: "queued",
            progress: 0,
            error: null,
            access_token: "subject-token",
            status_url: "/api/v1/tasks/subject-task",
            cancel_url: "/api/v1/tasks/subject-task/cancel",
          },
          { status: 202 },
        );
      }
      if (url.endsWith("/api/v1/tasks/subject-task")) {
        return Response.json({
          id: "subject-task",
          status: "completed",
          stage: "complete",
          progress: 1,
          error: null,
          preview_url: "/api/v1/tasks/subject-task/preview",
          download_url: "/api/v1/tasks/subject-task/download",
        });
      }
      if (url.endsWith("/preview")) {
        return new Response(new Blob(["preview"], { type: "image/png" }));
      }
      return Response.json({ status: "ok" });
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);

    fireEvent.change(screen.getByLabelText("Choose a photograph"), {
      target: { files: [imageFile()] },
    });
    fireEvent.click(screen.getByRole("button", { name: /protect selected subject/i }));
    fireEvent.change(screen.getByLabelText("Matting tile size"), {
      target: { value: "768" },
    });

    const frame = screen.getByRole("img", { name: /photo preview/i });
    vi.spyOn(frame, "getBoundingClientRect").mockReturnValue({
      x: 10,
      y: 20,
      left: 10,
      top: 20,
      right: 110,
      bottom: 220,
      width: 100,
      height: 200,
      toJSON: () => ({}),
    });
    fireEvent.click(frame, { clientX: 35, clientY: 80 });
    fireEvent.click(frame, { clientX: 80, clientY: 120 });
    expect(screen.getByText("Point 2")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /process photo/i }));

    await screen.findByRole("button", { name: /download full png/i }, { timeout: 3000 });
    const createCall = fetchMock.mock.calls.find(
      ([input, init]) => String(input).endsWith("/api/v1/tasks") && init?.method === "POST",
    );
    const body = createCall?.[1]?.body as FormData;
    const options = JSON.parse(String(body.get("options"))) as {
      subject_points: [number, number][];
      matte_tile_size: number;
    };
    expect(options.subject_points).toEqual([
      [0.25, 0.3],
      [0.7, 0.5],
    ]);
    expect(options.matte_tile_size).toBe(768);
  });

  it("caps subject protection at sixteen positive points", () => {
    vi.stubGlobal("fetch", vi.fn(async () => Response.json({ status: "ok" })));
    render(<App />);

    fireEvent.change(screen.getByLabelText("Choose a photograph"), {
      target: { files: [imageFile()] },
    });
    fireEvent.click(screen.getByRole("button", { name: /protect selected subject/i }));
    const frame = screen.getByRole("img", { name: /photo preview/i });
    vi.spyOn(frame, "getBoundingClientRect").mockReturnValue({
      x: 0,
      y: 0,
      left: 0,
      top: 0,
      right: 100,
      bottom: 100,
      width: 100,
      height: 100,
      toJSON: () => ({}),
    });
    for (let index = 0; index < 17; index += 1) {
      fireEvent.click(frame, { clientX: 20 + index, clientY: 30 + index });
    }

    expect(screen.getByText("Point 16")).toBeInTheDocument();
    expect(screen.queryByText("Point 17")).not.toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent("maximum of 16");
  });
});
