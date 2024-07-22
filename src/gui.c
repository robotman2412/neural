
// SPDX-License-Identifier: MIT

#include "neural.h"

#include <pax_gfx.h>
#include <SDL.h>

// PAX graphics context.
pax_buf_t     gfx;
// SDL window.
SDL_Window   *window;
// SDL renderer.
SDL_Renderer *renderer;



// Flush the contents of a buffer to the window.
void window_flush(SDL_Window *window, pax_buf_t *gfx) {
    static SDL_Texture *texture = NULL;
    static int          tw, th;
    SDL_Surface        *surface = SDL_GetWindowSurface(window);

    // Texture based update.
    if (texture && (tw != pax_buf_get_width(gfx) || th != pax_buf_get_height(gfx))) {
        SDL_DestroyTexture(texture);
        texture = NULL;
    }
    if (!texture) {
        tw      = pax_buf_get_width(gfx);
        th      = pax_buf_get_height(gfx);
        texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, tw, th);
    }
    if (texture) {
        SDL_Renderer *renderer = SDL_GetRenderer(window);
        void         *pixeldata;
        int           pitch;
        SDL_LockTexture(texture, NULL, &pixeldata, &pitch);
        memcpy(pixeldata, pax_buf_get_pixels(gfx), pax_buf_get_size(gfx));
        SDL_UnlockTexture(texture);
        SDL_RenderCopy(renderer, texture, NULL, NULL);
        SDL_RenderPresent(renderer);
    } else {
        puts(SDL_GetError());
    }
}

void gui_main() {
    // Create the SDL contexts.
    SDL_Init(SDL_INIT_VIDEO);
    int res = SDL_CreateWindowAndRenderer(400, 300, SDL_WINDOW_RESIZABLE, &window, &renderer);
    SDL_SetWindowMinimumSize(window, 100, 100);
    SDL_SetWindowTitle(window, "Neural");

    pax_buf_init(&gfx, NULL, 400, 300, PAX_BUF_32_8888ARGB);
    pax_background(&gfx, 0xff3f3f3f);
    window_flush(window, &gfx);

    SDL_Event event;
    while (1) {
        if (SDL_WaitEvent(&event)) {
            if (event.type == SDL_QUIT)
                break;
            if (event.type == SDL_WINDOWEVENT) {
                if (event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
                    pax_buf_destroy(&gfx);
                    int width, height;
                    SDL_GetWindowSize(window, &width, &height);
                    pax_buf_init(&gfx, NULL, width, height, PAX_BUF_32_8888ARGB);
                    pax_background(&gfx, 0xff7f7f7f);
                    window_flush(window, &gfx);
                }
            }
        } else {
            break;
        }
    }
}
