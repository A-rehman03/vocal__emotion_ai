import Vapi from "@vapi-ai/web";
import { VAPI_API_KEY, ASSISTANT_ID } from "../config";

export const vapi = new Vapi(VAPI_API_KEY);

export const startAssistant = async (overrides = {}) => {
    return await vapi.start(ASSISTANT_ID, overrides);
};

export const stopAssistant = () => {
    vapi.stop();
};
