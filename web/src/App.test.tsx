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
});
